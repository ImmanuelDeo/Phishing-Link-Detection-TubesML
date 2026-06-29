from src.pipeline.training_pipeline import TrainingPipeline

def main():
    config = {
        "test_size": 0.2,
        "random_state": 42,
        "correlation_threshold": 0.90,
        "k_best": 15,
        "cv_splits": 5,
        "invert_label": True,                                              
    }
    pipeline = TrainingPipeline(config=config)
    results_df = pipeline.run()
    return results_df
if __name__ == "__main__":
    main()