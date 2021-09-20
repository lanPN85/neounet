GPU=0
CONFIG=config/defaults.yml

DOCKER_RUN=docker run --rm -it\
	--shm-size=6g --network host\
	-e CUDA_VISIBLE_DEVICES=$(GPU)\
	-u $(shell id -u ${USER}):$(shell id -g ${USER})\
	-v $(shell pwd):/workspace\
	-v ${HOME}/.cache/torch:/cache/torch\
	lanpn85/neounet

shell:
	$(DOCKER_RUN) bash

image:
	docker build -t lanpn85/neounet .

image-gpu:
	docker build\
		--build-arg PYTORCH="torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html"\
		-t lanpn85/neounet .

TEST_OPT=""
test-psnd:
	$(DOCKER_RUN) python3 test_psnd.py $(TEST_OPT)

test-seg:
	$(DOCKER_RUN) python3 test_seg.py $(TEST_OPT)

train-seg:
	$(DOCKER_RUN) python3 train_seg.py -c $(CONFIG)

train-psnd:
	$(DOCKER_RUN) python3 train_psnd.py -c $(CONFIG)

.PHONY: image image-gpu train-seg train-psnd test-seg test-psnd
