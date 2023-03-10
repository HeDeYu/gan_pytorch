from gan_pytorch.gan.trainer import Trainer

output_dir = "../../../tests/output"

gan_trainer = Trainer(
    100,
    [64, 96],
    3,
    0.0002,
    0.0002,
    0.5,
    0.999,
    epochs=20000,
    batch_size=64,
    output_dir=output_dir,
)
gan_trainer.run()
