bl_info = {
    "name": "YOLO_FOTO",
    "author": "Victor Roza (com ajustes)",
    "version": (1, 5),
    "blender": (3, 6, 0),
    "location": "View3D > Painel Lateral > Fotografia",
    "description": "Gera câmeras ao redor de um objeto e renderiza imagens automaticamente, variando o cenário (imagens) e a posição de um Point Light, incluindo anotações COCO com segmentação.",
    "category": "Render",
}

import bpy
import os
import bpy_extras
import platform
import json
from bpy.props import PointerProperty, FloatProperty, StringProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup
from mathutils import Vector
from bpy.props import PointerProperty, FloatProperty, StringProperty, BoolProperty, IntProperty
from bpy.types import Panel, PropertyGroup
import math
import numpy as np
import cv2
import bpycv
import random
import bpy
import os
import random
from mathutils import Vector

# ----------------------------------------------------------------
#                        AJUSTAR CAMINHO
# ----------------------------------------------------------------
def ajustar_caminho(caminho):
    if platform.system() == "Windows":
        return caminho.replace("\\", "/")
    return caminho

# ----------------------------------------------------------------
#                  PROPRIEDADES PERSONALIZADAS
# ----------------------------------------------------------------
class FotoProperties(PropertyGroup):
    objeto: PointerProperty(
        name="Objeto",
        type=bpy.types.Object,
        description="Selecione o objeto geométrico",
    )
    distancia_focal: FloatProperty(
        name="Distância Focal",
        default=54.0,
        description="Defina a distância focal das câmeras",
    )
    apply_zoom: BoolProperty(
        name="Aplicar Zoom",
        default=False,
        description="Ativar ou desativar o zoom nas câmeras geradas",
    )
    zoom_level: FloatProperty(
        name="Nível de Zoom",
        default=1.0,
        min=0.1,
        max=5.0,
        description="Fator de zoom a ser aplicado. >1 para zoom in, <1 para zoom out",
    )
    num_annotations: IntProperty(
        name="Anotações por Câmera",
        default=1,
        min=1,
        description="Quantas amostras/anotações gerar por câmera",
    )
    output_dir: StringProperty(
        name="Diretório de Saída",
        default="//",
        subtype='DIR_PATH',
        description="Caminho para salvar as imagens e anotações",
    )
    imagem_cenario: StringProperty(
        name="Imagem para Cenário (Única)",
        default="",
        subtype='FILE_PATH',
        description="Caminho para a imagem que será usada no cenário",
    )
    imagens_cenario_dir: StringProperty(
        name="Pasta de Imagens para Cenário",
        default="",
        subtype='DIR_PATH',
        description="Caminho para a pasta com várias imagens de cenário",
    )
    limite_x: FloatProperty(
        name="Limite X",
        default=16.0,
        description="Limite máximo para a movimentação no eixo X",
    )
    limite_y: FloatProperty(
        name="Limite Y",
        default=16.0,
        description="Limite máximo para a movimentação no eixo Y",
    )
    limite_z_min: FloatProperty(
        name="Limite Z Mínimo",
        default=-15.0,
        description="Limite mínimo para a movimentação no eixo Z",
    )
    limite_z_max: FloatProperty(
        name="Limite Z Máximo",
        default=15.0,
        description="Limite máximo para a movimentação no eixo Z",
    )
    intensidade_luz: FloatProperty(
        name="Intensidade da Luz",
        default=1000.0,
        min=0.0,
        max=50000.0,
        description="Defina a potência da Point Light",
    )


# ----------------------------------------------------------------
#                      FUNÇÕES AUXILIARES
# ----------------------------------------------------------------
def get_rotation_in_degrees(obj):
    return tuple(math.degrees(angle) for angle in obj.rotation_euler)

def get_rotated_bbox(obj, camera, scene):
    """Retorna a bounding box 2D (em pixels) do obj, vista pela camera."""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    coords_2d = [
        bpy_extras.object_utils.world_to_camera_view(scene, camera, corner)
        for corner in bbox_corners
    ]

    min_x = min(coord.x for coord in coords_2d)
    max_x = max(coord.x for coord in coords_2d)
    min_y = min(coord.y for coord in coords_2d)
    max_y = max(coord.y for coord in coords_2d)

    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    min_x *= render_size[0]
    max_x *= render_size[0]

    min_y = (1 - min_y) * render_size[1]
    max_y = (1 - max_y) * render_size[1]

    x_min = min(min_x, max_x)
    x_max = max(min_x, max_x)
    y_min = min(min_y, max_y)
    y_max = max(min_y, max_y)

    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)

def mask_to_polygon(mask, keep_holes=False, eps_px=1.0, min_area=10):
    import cv2

    cnts, hier = cv2.findContours(
        mask,
        cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    segmentations = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        cnt = cv2.approxPolyDP(cnt, epsilon=eps_px, closed=True)
        if len(cnt) < 3:
            continue
        segmentations.append(cnt.reshape(-1).astype(int).tolist())

    if not keep_holes and segmentations:
        pass

    return segmentations

def save_coco_annotations_to_json(filename, annotations):
    """
    annotations: lista de dicionários com:
    {
      "name": str,
      "bbox": [x_min, y_min, width, height],
      "rotation": [rot_x, rot_y, rot_z],
      "segmentation": [[x1,y1,x2,y2,...], ...]
    }
    """
    data = {
        "annotations": annotations
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ----------------------------------------------------------------
#                     OPERADOR: GERAR CÂMERAS
# ----------------------------------------------------------------
class FOTO_OT_GerarCameras(Operator):
    bl_idname = "foto.gerar_cameras"
    bl_label = "Gerar Câmeras"
    bl_description = "Gera câmeras nos vértices do objeto selecionado"

    def execute(self, context):
        props = context.scene.foto_props
        objeto = props.objeto
        distancia_focal = props.distancia_focal
        apply_zoom = props.apply_zoom
        zoom_level = props.zoom_level

        if objeto and objeto.type == 'MESH':
            bpy.ops.object.mode_set(mode='OBJECT')
            objeto_center = objeto.location

            for obj in bpy.data.objects:
                if obj.type == 'CAMERA' and "Camera_Vertex_" in obj.name:
                    bpy.data.objects.remove(obj, do_unlink=True)

            for i, vertex in enumerate(objeto.data.vertices):
                vertex_global = objeto.matrix_world @ vertex.co
                bpy.ops.object.camera_add(location=vertex_global)
                camera = bpy.context.object
                direction = objeto_center - camera.location
                rot_quat = direction.to_track_quat('-Z', 'Y')
                camera.rotation_euler = rot_quat.to_euler()
                camera.data.lens = distancia_focal

                if apply_zoom:
                    camera.data.lens *= zoom_level

                camera.name = f"Camera_Vertex_{i+1}"
        else:
            self.report({'WARNING'}, "Por favor, selecione um objeto válido.")
        return {'FINISHED'}


# ----------------------------------------------------------------
#                     OPERADOR: CRIAR CENÁRIO
# ----------------------------------------------------------------
class FOTO_OT_CriarCenario(Operator):
    bl_idname = "foto.criar_cenario"
    bl_label = "Criar Cenário"
    bl_description = "Cria um cenário fechado ao redor do eixo central com uma imagem como textura"

    def execute(self, context):
        props = context.scene.foto_props
        imagem_path = ajustar_caminho(bpy.path.abspath(props.imagem_cenario))

        if not os.path.exists(imagem_path):
            self.report({'WARNING'}, f"Caminho da imagem inválido: {imagem_path}")
            return {'CANCELLED'}

        material = bpy.data.materials.new(name="CenarioMaterial")
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        
        tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(imagem_path)
        material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        cenario = bpy.context.object
        cenario.name = "CenarioQuadrao"
        cenario.scale = (20, 20, 20)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')

        if len(cenario.data.materials) > 0:
            cenario.data.materials[0] = material
        else:
            cenario.data.materials.append(material)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.cube_project(cube_size=1)
        bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, f"Cenário criado com sucesso usando a imagem: {imagem_path}")
        return {'FINISHED'}
    

def randomize_obj_texture(obj, textures_dir=None):
    """
    Mantém o material real em ~70 % dos renders e,
    nos demais, troca apenas a imagem do primeiro TEX_IMAGE
    conectado ao Base Color por outra textura plausível.
    """
    if random.random() < 0.7 or not textures_dir or not os.path.isdir(textures_dir):
        return

    mat = obj.active_material
    if not mat:
        return

    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not principled:
        return

    img_node = None
    for link in principled.inputs['Base Color'].links:
        if link.from_node.type == 'TEX_IMAGE':
            img_node = link.from_node
            break
    if img_node is None:
        img_node = nodes.new('ShaderNodeTexImage')
        img_node.location = principled.location.x - 400, principled.location.y
        links.new(img_node.outputs['Color'], principled.inputs['Base Color'])

    valid = [f for f in os.listdir(textures_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
             and any(k in f.lower() for k in ('metal', 'wood', 'paint', 'color', 'base'))]
    if not valid:
        valid = [f for f in os.listdir(textures_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    if valid:
        path = os.path.join(textures_dir, random.choice(valid))
        img_node.image = bpy.data.images.load(path)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def kelvin_to_rgb(temp_k):
    """
    Converte temperatura de cor (Kelvin) para RGB aproximado.
    """
    temp = temp_k / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        green = clamp(green, 0, 255)
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * math.log(temp - 10) - 305.0447927307
            blue = clamp(blue, 0, 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)
        red = clamp(red, 0, 255)
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        green = clamp(green, 0, 255)
        blue = 255
    return (red / 255.0, green / 255.0, blue / 255.0)


def setup_compositor(scene):
    """
    Configura o Compositor para adicionar ruído de sensor básico
    e preparar para pós-processamento.
    """
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    for n in nodes:
        nodes.remove(n)

    rl = nodes.new('CompositorNodeRLayers')
    mix = nodes.new('CompositorNodeMixRGB')
    mix.blend_type = 'ADD'
    mix.inputs[2].default_value = (0.03, 0.03, 0.03, 1.0)
    comp = nodes.new('CompositorNodeComposite')

    links.new(rl.outputs['Image'], mix.inputs[1])
    links.new(mix.outputs['Image'], comp.inputs['Image'])

# ----------------------------------------------------------------
#  OPERADOR: RENDERIZAR, ANOTAR  –  FUNDO + LUZ DINÂMICOS  **(fix var-clash)**
# ----------------------------------------------------------------
class FOTO_OT_RenderizarEAnotar(Operator):
    bl_idname = "foto.renderizar_e_anotar"
    bl_label = "Renderizar e Anotar"
    bl_description = ("Renderiza câmeras, salva anotações COCO e, a cada foto, "
                      "altera fundo, posição/potência/temperatura da Point Light")

    def execute(self, context):
        props   = context.scene.foto_props
        scene   = context.scene
        obj     = props.objeto
        tex_dir = bpy.path.abspath(props.imagens_cenario_dir)
        out_dir = bpy.path.abspath(props.output_dir)
        num_samples = int(props.num_annotations)

        bg_list = []
        if os.path.isdir(tex_dir):
            bg_list = [os.path.join(tex_dir, f) for f in os.listdir(tex_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

        cenario     = bpy.data.objects.get("CenarioQuadrao")
        bg_img_node = None
        if cenario and cenario.active_material:
            for node in cenario.active_material.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    bg_img_node = node
                    break
        # --------------------------------------

        if not obj:
            self.report({'WARNING'}, "Selecione um objeto antes de renderizar.")
            return {'CANCELLED'}
        os.makedirs(out_dir, exist_ok=True)

        point = bpy.data.objects.get("Point")
        if not point:
            bpy.ops.object.light_add(type='POINT', location=(0, 0, props.limite_z_max))
            point = bpy.context.object
            point.name = "Point"

        cams = [o for o in bpy.data.objects
                if o.type == 'CAMERA' and "Camera_Vertex_" in o.name]
        if not cams:
            self.report({'WARNING'}, "Nenhuma câmera de vértice encontrada.")
            return {'CANCELLED'}

        original_cam = scene.camera

        for cam in cams:
            scene.camera = cam
            zoom_modes = [False, True] if props.apply_zoom else [False]
            for use_zoom in zoom_modes:
                if use_zoom:
                    orig_lens = cam.data.lens
                    cam.data.lens *= props.zoom_level

                for idx in range(1, num_samples + 1):
                    # --------- fundo dinâmico ---------
                    if bg_list and bg_img_node:
                        img_path = random.choice(bg_list)
                        bg_img_node.image = bpy.data.images.load(img_path)

                    # --------- luz dinâmica -----------
                    point.location = (
                        random.uniform(-props.limite_x,  props.limite_x),
                        random.uniform(-props.limite_y,  props.limite_y),
                        random.uniform( props.limite_z_min, props.limite_z_max)
                    )
                    point.data.energy = random.uniform(5_000, 15_000)

                    base_k = 6500 if idx % 2 else 3000
                    kelvin = random.uniform(base_k - 500, base_k + 500)
                    point.data.color = kelvin_to_rgb(kelvin)
                    # ----------------------------------

                    randomize_obj_texture(obj, tex_dir)

                    suffix  = "_zoom" if use_zoom else "_normal"
                    imgname = f"{obj.name}_{cam.name}{suffix}_{idx:02d}.png"
                    scene.render.filepath = os.path.join(out_dir, imgname)
                    scene.render.image_settings.file_format = 'PNG'
                    bpy.ops.render.render(write_still=True)

                    data  = bpycv.render_data()
                    mask  = (data["inst"] == 1).astype(np.uint8)
                    ann   = [{
                        "name": obj.name,
                        "bbox": get_rotated_bbox(obj, cam, scene),
                        "rotation": get_rotation_in_degrees(obj),
                        "segmentation": mask_to_polygon(mask)
                    }]
                    annfile = f"{obj.name}_{cam.name}{suffix}_{idx:02d}_coco.json"
                    save_coco_annotations_to_json(
                        os.path.join(out_dir, annfile), ann
                    )

                if use_zoom:
                    cam.data.lens = orig_lens

        scene.camera = original_cam
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
        self.report({'INFO'}, "Renderização concluída com fundo e luz variáveis.")
        return {'FINISHED'}


# ----------------------------------------------------------------
#                     OPERADOR: ADICIONAR LUZ
# ----------------------------------------------------------------
class FOTO_OT_AdicionarLuz(Operator):
    bl_idname = "foto.adicionar_luz"
    bl_label = "Adicionar Luz"
    bl_description = "Adiciona uma Point Light e define sua intensidade"

    def execute(self, context):
        props = context.scene.foto_props

        intensidade = getattr(props, "intensidade_luz", 1000.0)
        if not isinstance(intensidade, (int, float)):
            self.report({'ERROR'}, "O valor da intensidade da luz não é numérico.")
            return {'CANCELLED'}

        point_light = bpy.data.objects.get("Point")
        if not point_light:
            bpy.ops.object.light_add(type='POINT', location=(0, 0, props.limite_z_max))
            point_light = bpy.context.object
            point_light.name = "Point"

        point_light.data.energy = float(intensidade)
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_recursive=True
        )
        self.report({'INFO'}, f"Luz adicionada com sucesso! Intensidade: {intensidade}")
        return {'FINISHED'}


FotoProperties.intensidade_luz = FloatProperty(
    name="Intensidade da Luz",
    default=1000.0,
    min=0.0,
    max=50000.0,
    description="Defina a potência da Point Light"
)


# ----------------------------------------------------------------
#                        PAINEL DE UI
# ----------------------------------------------------------------
class FOTO_PT_Painel(Panel):
    bl_label = "Fotografia Automática"
    bl_idname = "FOTO_PT_painel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'YOLO FOTO'

    def draw(self, context):
        layout = self.layout
        props = context.scene.foto_props

        layout.prop(props, "objeto")
        layout.prop(props, "distancia_focal")

        layout.separator()
        layout.prop(props, "apply_zoom")
        if props.apply_zoom:
            layout.prop(props, "zoom_level")
        layout.prop(props, "num_annotations")

        layout.separator()
        layout.prop(props, "output_dir")
        layout.prop(props, "imagem_cenario")
        layout.prop(props, "imagens_cenario_dir")

        layout.separator()
        layout.label(text="Limites da Luz:")
        layout.prop(props, "limite_x")
        layout.prop(props, "limite_y")
        layout.prop(props, "limite_z_min")
        layout.prop(props, "limite_z_max")

        layout.separator()
        layout.operator("foto.gerar_cameras")
        layout.operator("foto.criar_cenario")
        layout.operator("foto.renderizar_e_anotar")

        layout.label(text="Configuração da Luz:")
        layout.prop(props, "intensidade_luz")
        layout.operator("foto.adicionar_luz")

# ----------------------------------------------------------------
#                  REGISTRO / UNREGISTER
# ----------------------------------------------------------------
classes = [
    FotoProperties,
    FOTO_OT_GerarCameras,
    FOTO_OT_CriarCenario,
    FOTO_OT_RenderizarEAnotar,
    FOTO_OT_AdicionarLuz,
    FOTO_PT_Painel
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.foto_props = PointerProperty(type=FotoProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.foto_props

if __name__ == "__main__":
    register()
