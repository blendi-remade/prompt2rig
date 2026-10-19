"""Render an exported Prompt2Rig take (.glb) as a polished clip with Blender — no UI needed.

    blender -b -P tools/render_blender.py -- takes/<take>.glb [more.glb ...] [options]

Options
    --out PATH          output .mp4 (default: renders/<take>.mp4)
    --res WxH           resolution (default 1080x1920 portrait; e.g. 1920x1080)
    --samples N         EEVEE samples (default 64)
    --orbit DEG         camera arc across the clip (default 40)
    --loops N           play the animation N times (default 1)
    --side-by-side      also write <out>_compare.mp4: source clip | render (needs ffmpeg + the take's .json)
    --preview           quarter resolution, 8 samples — fast look-dev

Several .glb files are placed side by side (e.g. the same take exported for different characters).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import bpy
from mathutils import Vector

FPS = 30


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="render_blender.py")
    ap.add_argument("glbs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--res", default="1080x1920")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--orbit", type=float, default=40.0)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--preview", action="store_true")
    return ap.parse_args(argv)


def iter_fcurves(action):
    """F-curves of an action — Blender 4.4+ keeps them in layers/strips/channelbags."""
    if hasattr(action, "layers") and action.layers:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    yield from bag.fcurves
    elif hasattr(action, "fcurves"):
        yield from action.fcurves


# ---- scene -------------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    return scene


def import_character(path: str, x: float):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    for o in new:
        if o.parent is None:
            o.location.x += x
    for o in new:  # glTF materials can come in with backface culling / blend modes we don't want
        if o.type == "MESH":
            for slot in o.material_slots:
                if slot.material:
                    slot.material.use_backface_culling = False
    action = arm.animation_data.action if arm.animation_data else None
    end = int(math.ceil(action.frame_range[1])) if action else 1
    hips = next((b.name for b in arm.pose.bones if b.name.lower().endswith("hips")), None)
    return arm, hips, end


def make_material(name, color, roughness, specular=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    return m


def build_cyclorama():
    """Seamless studio floor curving up into a back wall (no visible horizon)."""
    import bmesh

    # (y, z) profile: flat floor → quarter-circle cove (r = 4 m) → vertical wall.
    cove = [(6.0 + 4 * math.sin(a), 4 - 4 * math.cos(a)) for a in (i * math.pi / 24 for i in range(13))]
    profile = [(-40.0, 0.0)] + cove + [(10.0, 30.0)]
    mesh = bpy.data.meshes.new("Cyc")
    bm = bmesh.new()
    rows = []
    for y, z in profile:
        rows.append((bm.verts.new((-40, y, z)), bm.verts.new((40, y, z))))
    for (a0, a1), (b0, b1) in zip(rows, rows[1:]):
        bm.faces.new((a0, a1, b1, b0))
    bm.to_mesh(mesh); bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("Cyc", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("Studio", (0.022, 0.024, 0.028), 0.28, 0.6))
    return obj


def add_area_light(name, loc, target, energy, color, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color, data.size = energy, color, size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_world(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    # Camera sees a near-black studio; reflections/lighting get a dim studio HDRI.
    hdri = nt.nodes.new("ShaderNodeTexEnvironment")
    path = os.path.join(bpy.utils.resource_path("LOCAL"), "datafiles", "studiolights", "world", "studio.exr")
    if os.path.exists(path):
        hdri.image = bpy.data.images.load(path)
    env = nt.nodes.new("ShaderNodeBackground"); env.inputs["Strength"].default_value = 0.35
    dark = nt.nodes.new("ShaderNodeBackground"); dark.inputs["Color"].default_value = (0.012, 0.013, 0.016, 1)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(hdri.outputs["Color"], env.inputs["Color"])
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(env.outputs["Background"], mix.inputs[1])
    nt.links.new(dark.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def setup_camera(scene, focus_arm, hips, width, end, orbit_deg, aspect):
    # Focus point follows the lead character's hips horizontally, at chest height.
    focus = bpy.data.objects.new("Focus", None)
    bpy.context.collection.objects.link(focus)
    focus.location = (0, 0, 1.0)
    if focus_arm and hips:
        c = focus.constraints.new("COPY_LOCATION")
        c.target, c.subtarget, c.use_z = focus_arm, hips, False

    pivot = bpy.data.objects.new("Pivot", None)
    bpy.context.collection.objects.link(pivot)
    c = pivot.constraints.new("COPY_LOCATION"); c.target = focus

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 45
    cam_data.sensor_fit = "VERTICAL"; cam_data.sensor_height = 24
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.parent = pivot
    scene.camera = cam
    t = cam.constraints.new("TRACK_TO"); t.target = focus; t.track_axis = "TRACK_NEGATIVE_Z"; t.up_axis = "UP_Y"

    # Distance that fits ~2.2 m tall and `width` m wide.
    v_half = math.atan(12 / cam_data.lens)
    h_half = math.atan(math.tan(v_half) * aspect)
    dist = max(1.1 / math.tan(v_half), (width / 2 + 0.35) / math.tan(h_half)) * 0.92
    # glTF +Z (toward the web camera) imports as Blender -Y: the camera sits on -Y.
    for frame, k_az, k_d, z in ((1, -0.5, 1.08, 1.05), (end, 0.5, 0.94, 1.25)):
        pivot.rotation_euler = (0, 0, math.radians(orbit_deg) * k_az)
        cam.location = (0, -dist * k_d, z)
        pivot.keyframe_insert("rotation_euler", frame=frame)
        cam.keyframe_insert("location", frame=frame)
    for obj in (pivot, cam):  # smooth ease in/out
        if obj.animation_data and obj.animation_data.action:
            for fc in iter_fcurves(obj.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"


def setup_render(scene, args, out):
    w, h = (int(v) for v in args.res.lower().split("x"))
    scale = 0.25 if args.preview else 1.0
    r = scene.render
    r.engine = "BLENDER_EEVEE"
    r.resolution_x, r.resolution_y = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    r.resolution_percentage = 100
    ee = scene.eevee
    ee.taa_render_samples = 8 if args.preview else args.samples
    if hasattr(ee, "use_raytracing"):
        ee.use_raytracing = True
    r.use_motion_blur = False  # EEVEE 5.2 crashes (access violation) with motion blur on skinned glTF in -b mode
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    im = r.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "VIDEO"
    im.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.filepath = out
    return w / h


def side_by_side(out, glb):
    meta_path = os.path.splitext(glb)[0] + ".json"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(meta_path):
        print("[render] side-by-side skipped (needs ffmpeg and the take's .json)")
        return
    src = json.load(open(meta_path, encoding="utf-8")).get("source")
    if not src:
        print("[render] side-by-side skipped (take has no source clip URL)")
        return
    cmp_out = os.path.splitext(out)[0] + "_compare.mp4"
    flt = ("[0:v]fps=30,scale=-2:1920,setsar=1[a];[1:v]scale=-2:1920,setsar=1[b];"
           "[a][b]hstack=inputs=2,scale=-2:1080[v]")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-i", out, "-filter_complex", flt,
                    "-map", "[v]", "-shortest", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", cmp_out], check=True)
    print(f"[render] wrote {cmp_out}")


def main():
    args = parse_args()
    scene = reset_scene()
    lead = None
    n = len(args.glbs)
    spacing = 1.5
    end = 1
    for i, glb in enumerate(args.glbs):
        arm, hips, e = import_character(glb, (i - (n - 1) / 2) * spacing)
        end = max(end, e)
        if i == (n - 1) // 2 or lead is None:
            lead = (arm, hips)
    total = end * max(1, args.loops)
    scene.frame_start, scene.frame_end = 1, total
    if args.loops > 1:  # repeat each armature action
        for o in bpy.data.objects:
            if o.type == "ARMATURE" and o.animation_data and o.animation_data.action:
                for fc in iter_fcurves(o.animation_data.action):
                    fc.modifiers.new("CYCLES")

    build_cyclorama()
    setup_world(scene)
    # Three-point lighting: warm key, cool rim, soft fill.
    add_area_light("Key", (-3.0, -3.5, 4.5), (0, 0, 1), 900, (1.0, 0.94, 0.86), 3.0)
    add_area_light("Rim", (3.0, 3.0, 3.8), (0, 0, 1.3), 1100, (0.62, 0.76, 1.0), 2.0)
    add_area_light("Fill", (3.5, -4.0, 1.6), (0, 0, 1), 220, (1.0, 1.0, 1.0), 4.0)

    out = os.path.abspath(args.out or os.path.join("renders", os.path.splitext(os.path.basename(args.glbs[0]))[0] + ".mp4"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    aspect = setup_render(scene, args, out)
    setup_camera(scene, lead[0], lead[1], spacing * (n - 1) + 1.0, total, args.orbit, aspect)

    print(f"[render] {n} character(s), {total} frames @ {FPS}fps → {out}")
    bpy.ops.render.render(animation=True)
    print(f"[render] wrote {out}")
    if args.side_by_side:
        side_by_side(out, args.glbs[0])


main()
