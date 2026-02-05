import os
import yaml
from accelerate import Accelerator
from easydict import EasyDict
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 你之前能用的镜像
from huggingface_hub import snapshot_download



# 加载配置文件
config = EasyDict(
    yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
)

snapshot_download(
    repo_id="facebook/dinov2-base",
    local_dir=config.sr.vfm.local_dir,
    token=os.environ.get("HF_TOKEN", None),  # 或者直接写死 token
    resume_download=True,
)
print("done")