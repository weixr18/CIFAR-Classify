import argparse

from test import Tester, SetTester
from train import Trainer
#from predict import Predictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="Train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    mode = args.mode

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

        }

        valid_rate = 0.1
        use_cuda = True

        trainer = Trainer()
        module_save_dir = './save/'
        trainer.setup(module_save_dir=module_save_dir,
                      valid_rate=valid_rate,
                      hyper_params=hyper_parameters,
                      use_cuda=use_cuda,
                      PRETRAINED=False)
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

    elif mode == "Test":

        print("Preparing...")
        hyper_parameters = {
            "batch_size": 64,
            "threads": 0,
            "input_size": (224, 224),
        }

        use_cuda = True
        TTA = False
        SHOW_PIC = False

        module_names = [
            "resnet-20201209195322-epoch-20-validacc-0.41752373417721517",
            "resnet-20201209200052-epoch-40-validacc-0.46143196202531644",
            "resnet-20201209200825-epoch-60-validacc-0.4549050632911392",
            "resnet-20201209201537-epoch-80-validacc-0.4452136075949367",
            "resnet-20201209202328-epoch-100-validacc-0.4489715189873418",
            "resnet-20201209203135-epoch-120-validacc-0.457871835443038",
            "resnet-20201209203922-epoch-140-validacc-0.4584651898734177",
            "resnet-20201209203922-epoch-140-validacc-0.4584651898734177",
        ]

        for name in module_names:

            module_path = "./save/" + name + ".pth"
            tester = SetTester(
                module_path=module_path,
                hyper_params=hyper_parameters,
                use_cuda=use_cuda
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

    """
    elif mode == "Demo":
        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
            "TTA_KERNEL_SIZE": (6, 6),
            "BG_KERNEL_SIZE": (8, 8),
            "DILATE_ITERATIONS": 10,
            "BIN_THRESHOLD": 0.60,
        }
        
   
        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_demo/"
        mask_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_demo_mask/"
        tmp_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/_tmp/_test/"
        test_rate = 1
        use_exist_dataset = False

        for e in range(50, 100, 50):

            module_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200524epoch-2000.pth"

            tester = Tester(
                module_path=module_path,
                cell_dir=cell_dir,
                mask_dir=mask_dir,
                tmp_dir=tmp_dir,
                exist_res_dir=exist_res_dir,
                hyper_params=hyper_parameters,
                use_cuda=use_cuda,
                use_exist_dataset=use_exist_dataset,
                test_rate=test_rate,
                USE_EXIST_RES=USE_EXIST_RES
            )
            import time
            tic = time.time()
            test_acc = tester.test(
                SHOW_PIC=SHOW_PIC,
                TTA=TTA
            )
            toc = time.time()
            print("module:", module_path.split('/')[-1])
            print("test accuracy:", test_acc)
       
    elif mode == "Predict":

        hyper_parameters = {
            "batch_size": 1,
            "threads": 0,
            "TTA_KERNEL_SIZE": (6, 6),
            "BG_KERNEL_SIZE": (8, 8),
            "DILATE_ITERATIONS": 10,
            "BIN_THRESHOLD": 0.6,
        }
        cell_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test/"
        model_path = "D:/Machine_Learning/Codes/CellSegment/save/unet-20200524epoch-2000.pth"
        save_dir = "D:/Machine_Learning/Codes/CellSegment/supplementary/dataset1/test_RES/"
        use_cuda = True
        TTA = False

        predictor = Predictor(model_path=model_path,
                              cell_dir=cell_dir,
                              save_dir=save_dir,
                              hyper_params=hyper_parameters,
                              use_cuda=use_cuda
                              )

        predictor.predict(TTA=TTA)
        pass
    
    """
    pass
