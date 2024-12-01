from faker import Faker
import webdataset as wds


def generate_run_name() -> str:
    """Generate a unique run name"""
    fake = Faker()
    return f"{fake.name().split()[0]}_{fake.color_name()}".lower()


def setup_dataloaders(train_dataset, val_datasets, config):
    """Setup train and validation dataloaders

    Args:
        train_dataset: Training dataset
        val_datasets: List of validation datasets
        config: TrainerConfig object

    Returns:
        tuple: (train_loader, val_loaders)
    """
    # Training loader setup
    train_loader = (
        wds.WebLoader(
            train_dataset,
            num_workers=config.num_workers,
            batch_size=None,
            persistent_workers=config.num_workers > 0,
        )
        .unbatched()
        .shuffle(1024)
        .batched(config.batch_size)
        .with_length(config.iterations)
    )

    # Validation loader setup
    val_loaders = []
    for val_ds in val_datasets:
        # Get dataset length
        try:
            # Try to get length directly if available
            dataset_length = len(val_ds)
        except Exception:
            # If length not available, try to count samples
            temp_loader = wds.WebLoader(val_ds).unbatched()
            dataset_length = sum(1 for _ in temp_loader)

        # Calculate number of batches
        num_batches = dataset_length // config.batch_size
        if dataset_length % config.batch_size != 0:
            num_batches += 1  # Add one more batch for remaining samples

        # Create validation loader
        val_loader = (
            wds.WebLoader(
                val_ds,
                num_workers=config.num_workers,
                batch_size=None,
                persistent_workers=config.num_workers > 0,
            )
            .unbatched()
            .batched(config.batch_size)
            .with_length(num_batches)  # Set length to number of batches
        )
        val_loaders.append(val_loader)

    return train_loader, val_loaders
