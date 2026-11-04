"""Author an ordinary, inspectable ComfyUI graph; no processing happens here."""

from .detailer import BBOX, CROP, FaceDetect, RegionSelect


def build_detailer_prompt(image, model, clip, vae, positive, negative, detector, sam=None, *, sampler="SamplerCustom", prefix="d_", **settings):
    graph = {}

    def add(name, kind, **inputs):
        identifier = prefix + name
        graph[identifier] = {"class_type": kind, "inputs": inputs}
        return lambda slot=0: [identifier, slot]

    def constant(name, kind, default):
        return add(name, kind, value=settings.get(name, default))

    feather = constant("feather", "PrimitiveInt", 5)
    noise_feather = constant("noise_mask_feather", "PrimitiveInt", 20)
    noise_mask = constant("noise_mask", "PrimitiveBoolean", True)
    inpaint = constant("inpaint_model", "PrimitiveBoolean", False)
    steps = constant("steps", "PrimitiveInt", 20)
    cfg = constant("cfg", "PrimitiveFloat", 8.0)
    denoise = constant("denoise", "PrimitiveFloat", 0.5)
    cycles = constant("cycles", "PrimitiveInt", 1)
    split_images = add("images", "ImpactImageBatchToImageList", image=image)
    outer = add("image_loop", "StartLoop", mode="List", **{"mode.list": split_images()}, cache_iterations=False)
    detect = add("detect", "MinFaceDetect", image=outer(3), detector=detector,
                 threshold=settings.get("bbox_threshold", 0.5), dilation=settings.get("bbox_dilation", 10),
                 crop_factor=settings.get("bbox_crop_factor", 3.0), drop_size=settings.get("drop_size", 10), image_index=outer())
    detection_slots = {output.id: index for index, output in enumerate(FaceDetect.define_schema().outputs)}
    columns = {name: detect(index) for name, index in detection_slots.items() if name not in ("count", "preview", "report")}
    mask_inputs = {"image": outer(3), **columns, "use_sam": settings.get("use_sam", False),
                   "detection_hint": settings.get("sam_detection_hint", "center-1"), "dilation": settings.get("sam_dilation", 0),
                   "threshold": settings.get("sam_threshold", 0.93), "bbox_expansion": settings.get("sam_bbox_expansion", 0),
                   "mask_hint_threshold": settings.get("sam_mask_hint_threshold", 0.7),
                   "mask_hint_use_negative": settings.get("sam_mask_hint_use_negative", "False"), "image_index": outer()}
    if sam is not None:
        mask_inputs["sam_model"] = sam
    masks = add("masks", "MinFaceMask", **mask_inputs)
    columns["masks"] = masks()
    plan = add("wildcard_plan", "MinDetailerWildcardPlan", **{name: columns[name] for name in ("region_index", "labels", "bbox_dtype", *BBOX, "masks")},
               wildcard=settings.get("wildcard", ""), seed=settings.get("seed", 0), image_index=outer())
    loop = add("region_loop", "StartLoop", mode="simple", **{"mode.num_iterations": plan(5)}, cache_iterations=False,
               parent_iteration=outer(), initial_iteration_value=outer(3))
    select = add("select", "MinFaceSelect", image=loop(4), **columns, process_index=loop(), order=plan(), seeds=plan(1), texts=plan(2), enabled=plan(3),
                 selected_indices=settings.get("selected_indices", "all"), preview_only=settings.get("preview_only", False), image_index=outer())
    selected_slots = {output.id: index for index, output in enumerate(RegionSelect.define_schema().outputs)}
    xy = {name: select(selected_slots[name]) for name in (*BBOX, *CROP)}
    patched_model = add("diffusion_model", "MinDetailerModel", model=model, noise_mask_feather=noise_feather())
    positive_crop = add("positive_crop", "MinCropConditioning", conditioning=positive, image=loop(4), **{name: xy[name] for name in CROP})
    negative_crop = add("negative_crop", "MinCropConditioning", conditioning=negative, image=loop(4), **{name: xy[name] for name in CROP})
    wildcard = add("wildcard_encode", "MinDetailerWildcardEncode", model=patched_model(), clip=clip, wildcard=select(selected_slots["wildcard"]))
    concatenate = add("concat", "ConditioningConcat", conditioning_to=positive_crop(), conditioning_from=wildcard(2))
    concat_choice = add("concat_choice", "MinMux2x1", select=plan(4), input0=wildcard(3), input1=concatenate())
    positive_choice = add("positive_choice", "MinMux2x1", select=select(selected_slots["has_wildcard"]), input0=positive_crop(), input1=concat_choice())
    crop = add("crop", "MinFaceCropUpscale", image=loop(4), mask=select(selected_slots["mask"]), **xy,
               guide_size=settings.get("guide_size", 512), guide_size_for=settings.get("guide_size_for", True),
               max_size=settings.get("max_size", 1024), feather=feather(), noise_mask_feather=noise_feather(), noise_mask=noise_mask(),
               force_inpaint=settings.get("force_inpaint", True), bbox_dtype=select(selected_slots["bbox_dtype"]),
               region_index=select(selected_slots["region_index"]), image_index=outer(), label=select(selected_slots["label"]))
    encoded = add("encode", "VAEEncode", pixels=crop(), vae=vae)
    masked_latent = add("latent_mask", "MinDetailerNoiseMask", samples=encoded(), mask=crop(3), enabled=noise_mask())
    inpainted = add("inpaint", "InpaintModelConditioning", positive=positive_choice(), negative=negative_crop(), vae=vae,
                    pixels=crop(), mask=crop(3), noise_mask=noise_mask())
    # ComfyMathExpression produces INT flags for the existing generic MUX nodes.
    inpaint_index = add("inpaint_index", "ComfyMathExpression", expression="int(a)", **{"values.a": inpaint()})
    latent_choice = add("latent_choice", "MinMux2x1", select=inpaint_index(1), input0=masked_latent(), input1=inpainted(2))
    positive_inpaint = add("positive_inpaint", "MinMux2x1", select=inpaint_index(1), input0=positive_choice(), input1=inpainted())
    negative_inpaint = add("negative_inpaint", "MinMux2x1", select=inpaint_index(1), input0=negative_crop(), input1=inpainted(1))
    cycle = add("cycle_loop", "StartLoop", mode="simple", **{"mode.num_iterations": cycles()}, cache_iterations=False,
                parent_iteration=loop(), initial_iteration_value=latent_choice())
    cycle_seed = add("cycle_seed", "ComfyMathExpression", expression="a + b", **{"values.a": select(selected_slots["seed"]), "values.b": cycle()})
    if sampler == "KSampler":
        sampled = add("sampler", "KSampler", model=wildcard(), seed=cycle_seed(1), steps=steps(), cfg=cfg(),
                      sampler_name=settings.get("sampler_name", "euler"), scheduler=settings.get("scheduler", "simple"),
                      positive=positive_inpaint(), negative=negative_inpaint(), latent_image=cycle(4), denoise=denoise())
        sampled_latent = sampled()
    else:
        schedule = add("schedule", "MinDetailerSchedule", model=wildcard(), sampler_name=settings.get("sampler_name", "euler"),
                       scheduler=settings.get("scheduler", "simple"), steps=steps(), denoise=denoise())
        sampled = add("sampler", "SamplerCustom", model=wildcard(), add_noise=True, noise_seed=cycle_seed(1), cfg=cfg(),
                      positive=positive_inpaint(), negative=negative_inpaint(), sampler=schedule(1), sigmas=schedule(), latent_image=cycle(4))
        sampled_latent = sampled(1)
    cycle_end = add("cycle_end", "EndLoop", output_value=sampled_latent, next_iteration_value=sampled_latent, accumulate=False)
    decoded = add("decode", "VAEDecode", samples=cycle_end(), vae=vae)
    composite = add("composite", "MinFaceComposite", image=loop(4), region=decoded(), blend_mask=crop(2), **{name: xy[name] for name in CROP},
                    region_index=select(selected_slots["region_index"]), image_index=outer(), label=select(selected_slots["label"]))
    sample_choice = add("sample_choice", "MinMux2x1", select=select(selected_slots["sample_enabled"]), input0=loop(4), input1=composite())
    size_choice = add("size_choice", "MinMux2x1", select=crop(4), input0=loop(4), input1=sample_choice())
    selected_choice = add("selected_choice", "MinMux2x1", select=select(), input0=loop(4), input1=size_choice())
    end = add("region_end", "EndLoop", output_value=selected_choice(), next_iteration_value=selected_choice(), accumulate=False)
    finalized = add("finalize", "MinDetailerFinalize", image=end(), region_count=detect())
    outer_end = add("image_end", "EndLoop", output_value=finalized(), accumulate=True)
    result = add("result", "ImageListToImageBatch", images=outer_end())
    return graph, result()
