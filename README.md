# cs231n-adni
Course project for CS231N - Visualizing CNNs for Interpretable Alzheimer's Diagnosis Through Neuroimaging

Project members: Zi Ying (Kathy) Fan, Elissa Li, Darian Martos

# Viewer Guide
## Repo structure
Here is a summary of each directory:
* ckpt: checkpoint files from training (may include: trained weights, stats, config)
* code: starter code provided by Qingyu Zhao
* data: pre-processed ADNI dataset images
* figures: graphs and results from this project
* metadata: additional info attached to dataset, and AAL brain areas
* new_code: code developed for the purposes of this project
* reports: pdf versions of milestone and final reports

The directory of most interest should be new_code. Below is a summary of the most relevant files:
* baseline_vis.ipynb: runs baseline visualization methods over test set
* confounder_vis.ipynb: runs confounder-aware technique on top of baseline methods, over test set
* classifier.py: main code for training binary classification CNN model

The most important helper files are:
* data.py, dx.txt, fold.txt, subjects_idx.txt: files used for data-processing and partitioning into train, val, test sets
* glm.py: contains code for performing stats test with general linear model, in order to use the confounder-aware technique
* model.py: contains pytorch modules for the baseline classification model, as well as modified model that can be used with the confounder-aware technique
* interpretation.py: code for baseline visualization methods
* utils.py, vis_utils.py: various utility functions to calculate accuracies, plot saliency maps, etc.

The following files are deprecated and should not be of interest:
* bitmask.py
* glm.ipynb
* occlusion_backprop_vis.ipynb
* plot.py

# Dev Guide
## GCP
### Specs/How-to
Project name: CS231N-ADNI

Project id: big-coil-275222

VM instance: Compute Engine -> VM instances (note: you can pin Compute Engine to your left navbar for easy access)
* Current VM: adni-runner-vm, with external IP address 35.203.128.65. For VM specs, see setup under https://github.com/cs231n/gcloud
* To ssh into the VM: Click on SSH dropdown -> view gcloud command
* Note: if you need to edit the VM, you must stop it first

Billing: currently using Kathy's account ($300 free credits). Should be fine for awhile.

Misc:
* "Notifications" icon on upper right is helpful for indicating when an action is ready (ex finished start-up of VM, finished stopping VM, etc)
* If you run gcloud and are prompted `Generating public/private rsa key pair. Enter passphrase (empty for no passphrase):` just leave it empty.

### One-time setup
* Follow https://github.com/cs231n/gcloud "Sign up GCP for the first time" to create an account if you don't have one yet
* Under "configure your project" you do not need to create another project; but you might need to upgrade your account to the full one
* If you haven't claimed the $50 course credits either, <b>do not</b> claim it since it cannot be transferred
* You may have to complete the "request an increase in GPU quota" step
* Follow the "install gcloud command-line tools" step. 
  * Try starting up the VM, running the "first time setup script" and verification underneath to see the GPU in action.
  * Unclear: I had set 'cs231n' as the password... is this the password everyone uses to access Jupyter notebooks?
  * Try running `jupyter notebook` from the home directory to open `demo.ipynb`. Note that you will need the external IP here. The port that the notebook connects to should be indicated in the terminal.
  * Please stop the VM after you're doing trying things out


## TODO at end of project:
* release static IP address
