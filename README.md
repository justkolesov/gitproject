# kolesov.as
### Project of course "Program Engineering for ML".

#### Run the current branch as follow:
```bash
$ git checkout feature/library
``` 


#### Run script for training
```bash
$ python ./code/train.py --test_size 0.2 --model_path "model_save"
``` 
Necessary parameters are test_size and model path. The first parameter is responsible for size of test samples, whereas the second is responsible for file, where model will be saved. By defualt:
 `--model_path 0.2
 --test_path "model_save"` 
If the code is completed succesfully, there is an output string as "model is trained"

#### Run script for testing
```bash
$ python ./code/test.py --test_size 0.2 --model_path "model_save"
``` 






 

