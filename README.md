# all-mpnet-base-v2

This is a cog model for the `all-mpnet-base-v2` sentence-transformers embedding model. This embedding model is based on [MPNet](https://arxiv.org/abs/2004.09297) and fine-tuned on 1 billion sentence pairs (see [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#:~:text=seb.sbert.net-,Background,-The%20project%20aims) for details). 


## Fine-tuning

### Example with NLI and MNLI

Download NLI and MNLI datasets and write them to json format: 

```
cog run python scripts/download_example_data.py 
```

Run training: 

```
cog run python training/trainer.py --data_path ./datasets/nli.json --max_steps 100 
```
## How to setup the cog model

### Prerequisites

GPU machine. You'll need a Linux machine with an NVIDIA GPU attached and the NVIDIA Container Toolkit installed. If you don't already have access to a machine with a GPU, check out our guide to getting a GPU machine.

Docker. You'll be using the Cog command-line tool to build and push a model. Cog uses Docker to create containers for models.

### Step 0: Install Cog
First, install Cog:

sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog

### Step 1: Set up weights

From the root directory of this repo, run: 

```
chmod +x scripts/download_and_prepare_model.py
cog run python scripts/download_and_prepare_model.py --config model_setup.yaml 
```

### Step 2: Run the model

You can run the model locally to test it:

cog predict -i text="You may know a word by the company it keeps."


Make sure to specify "private" to keep the model private.

### Step 4: Configure the model to run on either GPU or CPU.

Replicate supports running models on CPU or a variety of GPUs. The default GPU type is a T4 and that may be sufficient; however, for maximal batch side and performance, you may want to consider more performant GPUs like A100s.

Alternatively, if you will only be encoding single documents and you want to minimze spend at the cost of latency, you can run this model on CPU. You'll observe higher latencies, but this may be acceptable for your use case.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

### Step 5: Push the model to Replicate
Log in to Replicate:

cog login
Push the contents of your current directory to Replicate, using the model name you specified in step 3:

cog push r8.im/username/modelname
Learn more about pushing models to Replicate.

### Step 6: Run the model on Replicate
Now that you've pushed the model to Replicate, you can run it from the website or with an API.

To use your model in the browser, go to your model page.

To use your model with an API, click on the "API" tab on your model page. You'll see commands to run the model with cURL, Python, etc.

To learn more about how to use Replicate, check out our documentation.