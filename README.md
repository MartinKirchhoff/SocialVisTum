#AspectVisTum - A Toolkit for Graph Visualization and Automatic Labeling of Aspect Clusters in Text Data#

The model used to extract aspects/topics is based on an existing approach that can be found here: [Unsupervised-Aspect-Extraction](https://github.com/BrambleXu/Unsupervised-Attention-Aspect-Extraction)

##Test the toolkit in the browser using GitHub Pages##
To test the toolkit with existing data use the following link: [AspectVisTum Demo](https://martinkirchhoff.github.io/AspectVisTum/)

##Test the toolkit locally using the demo##
1. Download the folder visualization-demo from the repository

2. Open a terminal window in the folder visualization-demo.

3. Use "conda create -n visualization_demo python=2.7“ to set up a new conda environment with the name "visualization_demo"

4. Activate the environment with "source activate visualization_demo"

5. To show the visualization in the browser, use „python display_visualization.py“.  
 If an error occurs, this might be because another application is already using the port 9000.   
In this case, specify another port by using "python display_visualization.py --port &lt;port&gt;" instead.

##Test the toolkit with new data or parameters##

1. Download the folder visualization-project from the repository.

2. Open a terminal window in the folder visualization-project.

3. Use "conda env create -f environment.yml" to create a new environment with the name "visualization_env", which contains the required packages.

4. Activate the environment with "source activate visualization_env"

5. Download the pre-trained GloVe embeddings with 6B tokens (glove.6B.zip) from https://nlp.stanford.edu/projects/glove/

6. Extract the zip and put the whole folder (glove.6B) into the folder pretrained_embeddings.

7. Navigate to the datasets folder and copy your dataset as a subfolder in the directory.  The name of the subfolder will be used as &lt;domain&gt; of your dataset in the following commands.

8. To preprocess the data and finetune the embeddings use "python ./code/preprocess.py --domain-name &lt;domain&gt;"  
The parameter &lt;domain&gt; specifies the name of the folder that includes the data set.   
If you want to use the existing organic food data set use organic_food_preprocessed.

9. To extract topics and topic information and create the JSON file, use "python ./code/train.py --domain &lt;domain&gt; --conf &lt;conf&gt;--emb-path ./preprocessed_data/&lt;domain&gt;/glove_w2v_300_filtered.txt"
&lt;conf&gt; is used to create a seperate result folder for each time you train the model with different parameters. You can simply use 0 for the first time.
Moreover, you can adjust all relevant training parameters as explained in the Parameter Overview section. The results are saved in the folder /code/output_dir/&lt;domain&gt;/&lt;conf&gt;.

10. To show the visualization and labeling tool in the browser, use "python ./code/display_visualization.py --domain &lt;domain&gt; --conf &lt;conf&gt; --port &lt;port&gt;".  
&lt;port&gt; specifies the port where it is opened and is by default set to 9000.  
If the visualization does not show anything or the visualization shows the result of a previous setting, the port might already be in use. To get the correct result, simply use another port.

11. To calculate the F1 scores use "python ./code/evaluation.py --domain &lt;domain&gt; --conf &lt;conf&gt;". The results are saved in the folder /code/output_dir/&lt;domain&gt;/&lt;conf&gt; in the file results.

**Training Parameter Overview**

All the training parameters can be adjusted by adding the name of the parameter and its associated value to the "python ./code/train.py" command.   
Example: "python ./code/train.py --domain &lt;domain&gt; --conf &lt;conf&gt;--emb-path ./preprocessed_data/&lt;domain&gt;/glove_w2v_300_filtered.txt --num-topics 5 --num-words 30"

| Name                  | Explanation                                             | Type  | Default                            |
|-----------------------|---------------------------------------------------------|-------|------------------------------------|
| --domain              | Domain of the corpus                                    | str   | required                           |
| --conf                | Train configuration for the given domain                | str   | required                           |
| --emb-path            | Path to the word embedding file                         | str   | required                           |
| --num-topics          | Number of topics detected                               | int   | 20                                 |
| --vocab-size          | Vocabulary size. 0 means no limit                       | int   | 9000                               |
| --num-words           | Number of representative sentences displayed            | int   | 10                                 |
| --num-sentences       | Number of representative words displayed                | int   | 10                                 |
| --labeling-num-words  | Number of representative words used to generate labels  | int   | 25                                 |
| --batch-size          | Batch size used to train the model                      | int   | 64                                 |
| --epochs              | Number of epochs                                        | int   | 20                                 |
| --neg-size            | Number of negative instances used for training          | int   | 20                                 |
| --maxlen              | Maximum number of words in every sentence               | int   | 0                                  |
| --algorithm           | Optimization algorithm used (rmsprop, sgd, adam...)     | str   | "adam"                             |
| --fix-clusters        | Fix initial topic clusters ("yes" or "no")              | str   | "no"                               |
| --ortho-reg           | Weight of orthogonal regularization                     | float | 0.1                                |
| --probability-batches | Number of batches used to calculate topic probabilities | int   | Number of training examples / 5000 |
| --emb-dim             | Embeddings dimension                                    | int   | 300                                |
| --emb-type            | Type of word embedding to use                           | str   | "glove_finetuned"                  |
