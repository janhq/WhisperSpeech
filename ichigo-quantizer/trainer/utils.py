from faker import Faker
import webdataset as wds

def generate_run_name() -> str:
    """Generate a unique run name"""
    fake = Faker()
    return f"{fake.name().split()[0]}_{fake.color_name()}".lower()

def setup_dataloaders(train_dataset, val_datasets, config):
    """Setup train and validation dataloaders"""
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

    val_loaders = [
        wds.WebLoader(
            val_ds,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
        )
        for val_ds in val_datasets
    ]

    return train_loader, val_loaders