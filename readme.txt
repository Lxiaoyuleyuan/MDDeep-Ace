1. Environment Setup
First, you need to install the required packages specified in the requirements.txt file. Open the command line terminal and navigate to the directory where the requirements.txt file is located. Then, run the following command:
pip install -r requirements.txt

2. Running the Main Scripts
2.1 Running main_stage1.py
After the environment is set up, you can run the main_stage1.py file. Navigate to the directory containing the main_stage.py file in the command line terminal. Then, run the following command:
python main_stage1.py

2.2 Running main_stage2.py
Similarly, to run the main_stage2.py file, navigate to the directory where the main_stage2.py file is located in the command line terminal. Then, execute the following command:
python main_stage2.py

2.3 Running test.py
If you want to test the auc metrics of our model on different species, you can ignore the above two steps and directly run python test.py in the command line. Eventually, it will output whether the operating environment of the model is gpu or cpu, and also output the test auc results on the specified species. Meanwhile, modifying the specie_num variable in line 37 of the code can modify the species type

Note: if your runtime environment prompts errors, then there may be cuda version compatibility issues, you need to configure the environment according to your current computer cuda version, for the code function comments you can view the main_stage1 and main_stage2 and densenet code block comments