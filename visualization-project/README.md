1. Download the pre-trained GloVe embeddings with 6B tokens (glove.6B.zip) from https://nlp.stanford.edu/projects/glove/ 
2. Extract the zip and put the whole folder (glove.6B) to the folder pretrained_embeddings.
3. Navigate to the datasets folder and copy your dataset as a subfolder in the directory. The name of the subfolder will be used as <domain> of your dataset in the following commands.
4. Use "python ./code/preprocess.py --domain-name <domain>"
   This will use the pre-trained embeddings, finetune them on the data and do all necessary preprocessing steps.
5. Use "python ./code/train.py --domain <domain> --conf <conf> --emb-path ./preprocessed_data/<domain>/glove_w2v_300_filtered.txt"
   <conf> specifies the configuration name and is used to specify the name of the subfolder when you use the same data with multiple settings. You can simply use 0 for the first use. You only need to do this step once for a dataset. 
   Moreover, you can specify all the other parameters that are defined in the train.py class.
   The method will extract the topics and generate all additional topic information (topic words and sentences, topic occurrences, topic similarities, JSON file for visualization)
   The results can be found under /code/output_dir/<domain>/<conf>
6. Use "python ./code/display_visualization.py --domain <domain> --conf <conf> --port <port>" to show the visualization and open it in the browser.
   <port> specifies the port where it is opened and is by default set to 9000. If the visualization does not show or the visualization shows the result of a previous setting, the port might already be in use. To get the correct result, simply use another port.
7. To calculate the F1 scores use "python ./code/evaluation.py --domain <domain> --conf <conf>"
If you change the vocabulary size or the number of topics in the training, you also must specifiy these parameters during evaluation because an error will occur otherwise. To do so, use append "--vocab-size <vocab-size> --num-topics <num-topics>" to the command. The results are saved in the ouput directory under results.
   
   python ./code/display_visualization.py --domain organic_food_preprocessed --conf 0 --port 10000
   
python ./code/evaluation.py --domain organic_food_preprocessed --conf 0 --vocab-size 10000 --num-topics 5
   
MISSING source activate ...python ./code/preprocess.py --domain-name organic_food_preprocessed
python ./code/train.py --domain organic_food_preprocessed --conf 0 --emb-path ./preprocessed_data/organic_food_preprocessed/glove_w2v_300_filtered.txt --epochs 5 --vocab-size 10000 --num-topics 5