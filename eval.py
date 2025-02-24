from torchreid.data.datasets.image.anon_reid_tsh import AnonymizedReIDTSH
from torchreid.data.datasets.image.anon_reid_ntu import AnonymizedReIDNTU


def main():
    import torchreid

    # 1. Register your custom dataset
    torchreid.data.register_image_dataset("anonymized_reid", AnonymizedReIDNTU)

    # 2. Create data manager
    datamanager = torchreid.data.ImageDataManager(
        root="/lsdf/data/activity/NTU_RGBD/reid",
        sources="market1501",
        targets="anonymized_reid",  # Your dataset as target
        height=256,
        width=128,
        batch_size_test=100,
    )

    # 3. Create model
    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=datamanager.num_train_pids, loss="softmax")
    torchreid.utils.load_pretrained_weights(model, "osnet_ms_d_c.pth.tar")

    # 4. Initialize engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=None,  # No optimizer needed for testing only
    )

    # 5. Run evaluation only
    engine.run(
        save_dir="log/market_to_anonymized",
        test_only=True,
        dist_metric="cosine",  # Standard in ReID, often performs better than euclidean
        normalize_feature=True,  # Always True with cosine distance
        eval_freq=1,
        rerank=True,  # Re-ranking typically improves results
        visrank=True,  # Useful for analysis
        visrank_topk=10,
    )


if __name__ == "__main__":
    main()
