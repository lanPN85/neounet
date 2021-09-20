# U-Net for polyp detection

## Requirements
- Python 3.6+
- Docker
- Make

## Quickstart
Download training data to the `data/` folder or edit the configuration file to point to your data.

For non-Docker users:
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
./train_seg.py -c config/seg/demo.yml
```

For Docker users:
```bash
# Build the Docker image
make image-gpu  # Change to 'make image' if running on CPU

# Run demo
make train-seg CONFIG=config/seg/demo.yml
```

## Configuration guide
Configs are written in YAML files. See `config/defaults.yml` for a full example.

Modules like `dataset` and `model` are configured by specifying the Python class (`cls`) and keyword arguments (`kwargs`). The class is specified by its full Python path (loaded using `pydoc.locate()`). For example, this configuration:
```yaml
model:
  cls: polypnet.model.eff.AttnEfficientNetUnet
  kwargs:
    backbone_name: "efficientnet-b0"
```
is equivalent to creating a variable as:
```python
class_ = pydoc.locate("polypnet.model.eff.AttnEfficientNetUnet")
model = class_(backbone_name="efficientnet-b0")
```
