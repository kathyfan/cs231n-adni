# cs231n-adni
Course project for CS231N - investigating interpretability and visualization of CNNs on ADNI dataset.

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

### One-time setup
* Follow https://github.com/cs231n/gcloud "Sign up GCP for the first time" to create an account if you don't have one yet
* Under "configure your project" you do not need to create another project; but you might need to upgrade your account to the full one
* If you haven't claimed the $50 course credits either, <b>do not</b> claim it since it cannot be transferred
* You may have to complete the "request an increase in GPU quota" step
* Follow the "install gcloud command-line tools" step. Try starting up the VM, running the "first time setup script" and verification underneath to see the GPU in action.
  * Unclear: I had set 'cs231n' as the password... is this the password everyone uses?
  * Please stop the VM after you're doing trying things out




## TODO at end of project:
* release static IP address
