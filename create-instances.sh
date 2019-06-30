#!/bin/bash

for i in $(seq 1 6)
  do
  	echo "Creating worker number: " $i
		gcloud compute instances create spec-worker-$i \
		  --zone=us-central1-a\
		  --scopes=storage-rw \
		  --trace-email=ignacio-gcloud-storage@music-genre-244102.iam.gserviceaccount.com \
		  --machine-type=n1-standard-4 \
		  --verbosity=debug


		echo "Worker created"
		echo "Starting provisioning..."
		./provision-worker.sh $i

		echo "Machine " $i " ready."
done

