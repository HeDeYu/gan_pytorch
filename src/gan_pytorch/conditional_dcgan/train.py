from gan_pytorch.conditional_dcgan.trainer import Trainer

output_dir = "../../../tests/output"

gan_trainer = Trainer(
    10,
    100,
    1,
    32,
    0.0002,
    0.0002,
    0.5,
    0.999,
    epochs=20,
    batch_size=64,
    output_dir=output_dir,
    g_base_channel=128,
    d_base_channel=128,
)
gan_trainer.run()
