import torch


class LoraTraining(torch.nn.Module):
    def __init__(self, inference_model, gradient_accumulation_steps) -> None:
        super().__init__()

        self.inference_model = inference_model

        self.optimizer = None
        self.lr_scheduler = None

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = None

        self.calibrate = False
        self.run_optimizer = False

    def update_training_parameters(self, optimizer, lr_scheduler, training_args):
        assert self.gradient_accumulation_steps == training_args.gradient_accumulation_steps

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = training_args.max_grad_norm

    def forward(self, inputs):
        # Remove this once hybrid model supports multiple inputs
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4568
        x, y = inputs

        # some parts on server side
        outputs = self.inference_model(input_ids=x, labels=y)

        loss = outputs.loss
        loss = loss / self.gradient_accumulation_steps

        # Update gradients
        # We need to set requires grad to the loss manually because the inference model's last
        # step is the "lm_head" layer, which is detached from the graph by the hybrid model
        loss.requires_grad_(True)
        loss.backward()

        grad_norm = None
        if not self.calibrate and self.run_optimizer:
            assert self.optimizer is not None
            assert self.lr_scheduler is not None
            assert self.max_grad_norm is not None

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.inference_model.parameters(), max_norm=self.max_grad_norm, norm_type=2
            )

            self.optimizer.step()
            self.lr_scheduler.step()

            self.inference_model.zero_grad()

        # Clean gradients after calibration
        elif self.calibrate:
            self.inference_model.zero_grad()

        return (loss, grad_norm)

    def toggle_calibrate(self, enable: bool = True):
        self.calibrate = enable

    def toggle_run_optimizer(self, enable: bool = True):
        self.run_optimizer = enable
