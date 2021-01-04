"# Artifact_Removal_Segmentation_Models_CBCT" 


All necessary packages are installed with setup.py


Setup 
------

-   import anaconda.sh

-   Install anaconda on server, conda init

-   Create env: conda create -n ENVNAME python=3.7 , activate 

-   conda install -c conda-forge tmux

-   pip install guildai 

-   Guild check (-> note guild home) 

-   create tmux sessions:   tmux new -s session_name

-   Might to restart server to make installations functional

### For each run:

-   conda activate ENVNAME

-   tmux attach -t session_name

-   cd /bigstore/shannon/Code

-   Run code:

-   Check gpus with top or nvidia-smi

-   To run: CUDA_VISIBLE_DEVICES=X guild run + flags 

-   Example: CUDA_VISIBLE_DEVICES=1 guild run architecture='unet' batch_size=4 lr=0.001 it=b4

-   So iteration also has the purpose of giving the run a name

-   Then detach from session: Ctr+b, then d

-   (to delete session Ctr+b , then : , then kill-session)

### After run:

-   Reattach to session with tmux attach-session -t session_name (or number - find with tmux ls)

-   After running send guild runs to export

-   cd /bigstore/shannon/ReconstructionResults

-   bash send_scan.sh  (creates a folder in Examined with selected results and this is then sent to titan)

-   Enter name of run in /ReconstructionResults

-   Create directory in titan.ci.cwi.nl:/export/scratch1/jordi/shannon/results/ for the run 

-   Enter the titan directory when prompted

Guild instructions:

To view runs:

-   guild compare

-   Exit comparison table with 'q'

-   Delete runs: guild runs delete 1 (first run is deleted, or first 2 runs are deleted) 

-   Delete all terminated/error runs: guild runs delete --terminated / guild runs delete --error

When finished with all the experiments,  all guild runs can be sent to titan, so we have them together to compare if needed. 

-   Download runs to my laptop, and save in guild_home on my lapto

-   To view on my laptop: conda activate conebeam, guild compare

