ACR_REGISTRY="final_test_answer.azurecr.io"
ACR_USERNAME="karan36k@gmail.com"
ACR_PASSWORD="**********"

docker login $ACR_REGISTRY -u $ACR_USERNAME -p $ACR_PASSWORD

docker tag dependency_image:latest $ACR_REGISTRY/dependency_image:latest
docker push $ACR_REGISTRY/dependency_image:latest

docker tag final_image:latest $ACR_REGISTRY/final_image:latest
docker push $ACR_REGISTRY/final_image:latest

docker logout $ACR_REGISTRY
