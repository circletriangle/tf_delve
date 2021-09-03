#!/bin/bash
# sync changes to .py files in '.' to docker container named volume 
# mounted on dir 'dest' for tf-gpu execution.
# -> to be kept running attached to a terminal while editing code


while inotifywait -e modify,create,delete,move /home/leo/Desktop/Thesis/New_Repo/; do
	printf "##########\n# SYNCING\n##########\n"
	rsync -v /home/leo/Desktop/Thesis/New_Repo/*.py /home/leo/Desktop/Thesis/mnt/bind_mount_tf_gpu/
done	
