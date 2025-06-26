
def get_datasets(P, training_type, config, fold, folds, outcome,
                 tile_px=256, tile_um=129):
    """
    Load train/val datasets, applying:
    - CV vs standard split
    - early stopping
    - binary‐outcome filter
    - optional cohort-2 / squamous / chemo-immuno filters from config
    """
    import pandas as pd

    # pull flags from your config
    use_cohort2     = config.get("USE_COHORT2_FILTER", False)
    use_all_mods    = config.get("FILTER_ALL_MODS", False)

    print("\n📂 Loading annotations from:", P.annotations)
    annotations_df = pd.read_csv(P.annotations)
    print("📊 Annotations shape:", annotations_df.shape)

    fold_col       = f"fold_{outcome}"
    early_stop_col = f"early_stopping_{outcome}"
    train_filter, val_filter = {}, {}

    # ─── your existing CV / standard logic ────────
    if training_type == "cross_validation":
        train_filter[fold_col] = [f for f in folds if f != fold]
        val_filter[fold_col]   = [fold]
        if early_stop_col in annotations_df:
            train_filter[early_stop_col] = "no"
            val_filter[fold_col] = [f for f in folds if f != fold]
            val_filter[early_stop_col]   = "yes"

    elif training_type == "standard":
        if early_stop_col in annotations_df:
            train_filter[f"dataset_{outcome}"] = "train"
            train_filter[early_stop_col]       = "no"
            val_filter[f"dataset_{outcome}"] = "train"
            val_filter[early_stop_col]         = "yes"
        else:
            train_filter[f"dataset_{outcome}"] = "train"
            val_filter[f"dataset_{outcome}"]   = "test"
    else:
        raise ValueError(f"Unknown training type: {training_type}")

    # ─── binary‐outcome filter ────────────────────
    for b in ["os_months_6","os_months_24","DCR","ORR"]:
        if outcome.lower() == b.lower():
            train_filter[outcome] = ["1.0","0.0"]
            val_filter[outcome]   = ["1.0","0.0"]

    # ─── NEW: apply cohort-2 if requested ────────
    if use_cohort2 and "COHORT_2" in annotations_df:
        print("🔖 Filtering COHORT_2 == 1")
        train_filter["COHORT_2"] ="1"
        val_filter["COHORT_2"]   = "1"

    # ─── NEW: apply “all modalities” filter if requested ─────────
    if use_all_mods and "HAS_ALL_MODALITIES" in annotations_df:
        print("🔖 Filtering HAS_ALL_MODALITIES == 1")
        train_filter["HAS_ALL_MODALITIES"] = "1"
        val_filter["HAS_ALL_MODALITIES"]   = "1"
    

    # ─── NEW: apply squamous filter if set to 0 or 1 ──────────────
    sq_flag = config.get("FILTER_SQUAMOUS", None)
    if sq_flag in {"0.0", "1.0"} and "NSCLC_HISTOLOGY_SQUAMOUS" in annotations_df:
        print(f"🔖 Filtering NSCLC_HISTOLOGY_SQUAMOUS == {sq_flag}")
        train_filter["NSCLC_HISTOLOGY_SQUAMOUS"] = [sq_flag]
        val_filter["NSCLC_HISTOLOGY_SQUAMOUS"]   = [sq_flag]

    # ─── NEW: apply chemo+immuno filter if set to 0 or 1 ──────────
    ci_flag = config.get("FILTER_CHEMO_IMMUNO", None)
    if ci_flag in (0, 1) and "IO_IOCT" in annotations_df:
        print(f"🔖 Filtering IO_IOCT == {ci_flag}")
        train_filter["IO_IOCT"] = [ci_flag]
        val_filter["IO_IOCT"]   = [ci_flag]

    # ─── NEW: apply PDL1_GROUP filter if set to "low" or "high" ──────────
    pdl1_flag = config.get("FILTER_PDL1", None)
    if pdl1_flag in ("low", "high") and "PDL1_GROUP" in annotations_df:
        print(f"🔖 Filtering PDL1_GROUP == {pdl1_flag}")
        train_filter["PDL1_GROUP"] = [pdl1_flag]
        val_filter["PDL1_GROUP"]   = [pdl1_flag]

    # ─── finalize & load ─────────────────────────
    print("\n🔎 FINAL train_filter:", train_filter)
    print("🔎 FINAL val_filter:  ", val_filter)

    train_ds = P.dataset(tile_px=tile_px, tile_um=tile_um, filters=train_filter)
    val_ds   = P.dataset(tile_px=tile_px, tile_um=tile_um, filters=val_filter)

    return train_ds, val_ds




 