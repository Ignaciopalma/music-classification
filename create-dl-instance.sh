echo "Creating Deep Learning Machine: "

ZONE=us-west1-b

gcloud compute instances create spec-training \
  --zone=$ZONE \
  --scopes=storage-rw \
  --trace-email=ignacio-gcloud-storage@music-genre-244102.iam.gserviceaccount.com \
  --machine-type=n1-standard-8 \
  --maintenance-policy TERMINATE --restart-on-failure \
  --accelerator=type=nvidia-tesla-k80 \
  --tags http-server,https-server \
  --image-family=tf-latest-gpu \
  --image-project=deeplearning-platform-release \
  --service-account=ignacio-gcloud-storage@music-genre-244102.iam.gserviceaccount.com \
  --metadata="install-nvidia-driver=True" \
  --verbosity=info

echo "DL machine created"
echo "Starting provisioning..."

gcloud compute scp ./training-requirements.txt ignaciopalma@spec-training:/home/ignaciopalma/ --zone=$ZONE
gcloud compute scp ./small_train_labels.csv ignaciopalma@spec-training:/home/ignaciopalma/small_train_labels.csv --zone=$ZONE

echo "Machine ready."

echo "ssh into instance.."

gcloud compute ssh spec-training --zone=$ZONE
