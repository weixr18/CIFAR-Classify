import argparse

from test import Tester
from train import Trainer
from predict import Predictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="Train")
    parser.add_argument("--task", "-t", type=int)
    parser.add_argument("--module_name")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mode = args.mode
    mission = args.task
    if (mission == None):
        print("Please input task number.")
        return

    if mode == "Train":
        print("Preparing...")
        hyper_parameters = {
            "input_size": (224, 224),
            "batch_size": 64,
            "threads": 0,

            "epochs": 2000,
            "epoch_lapse": 2,
            "epoch_save": 20,

            "learning_rate": 1e-3,
            "step_size": 20,
            "lr_gamma": 0.85,

            "optimizer": "Adam",
            "name_prefix": "CR+HF+RS"

        }

        valid_rate = 0.1
        use_cuda = True

        trainer = Trainer()
        module_save_dir = './save/'
        trainer.setup(module_save_dir=module_save_dir,
                      valid_rate=valid_rate,
                      hyper_params=hyper_parameters,
                      use_cuda=use_cuda,
                      PRETRAINED=False,
                      mission=mission)
        print("Model ready.")

        trainer.train()
        trainer.save_module()

    elif mode == "Validate":

        print("Preparing...")
        hyper_parameters = {
            "batch_size": 64,
            "threads": 0,
            "input_size": (32, 32),
        }

        test_rate = 0.99
        use_cuda = True
        TTA = False
        SHOW_PIC = False

        module_names = [
            "resnet-20201209203922-epoch-140-validacc-0.4584651898734177",
        ]

        for name in module_names:

            module_path = "./save/" + name + ".pth"
            tester = Tester(
                module_path=module_path,
                hyper_params=hyper_parameters,
                use_cuda=use_cuda,
                test_rate=test_rate
            )
            print("Model ready.")

            import time
            tic = time.time()
            test_acc = tester.test(
                SHOW_PIC=SHOW_PIC,
                TTA=TTA
            )
            toc = time.time()

            print("module:", module_path.split('/')[-1])
            print("test accuracy:", test_acc, "time:", toc - tic)

        pass

    elif mode == "Predict":

        print("Preparing...")
        hyper_parameters = {
            "batch_size": 64,
            "threads": 0,
            "input_size": (224, 224),
        }

        module_name = args.module_name
        if (module_name == None):
            print("Please input module name.")
            return
        save_dir = "res/"
        use_cuda = True
        TTA = False

        name = module_name
        model_path = "./save/" + name + ".pth"
        predictor = Predictor(model_path=model_path,
                              save_dir=save_dir,
                              hyper_params=hyper_parameters,
                              use_cuda=use_cuda,
                              mission=mission)

        print("Model ready.")
        predictor.predict(TTA=TTA)

        pass

    pass


if __name__ == "__main__":
    main()
