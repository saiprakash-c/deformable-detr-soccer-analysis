import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import urllib3
import certifi
import ssl

# Create an SSLContext with the desired SSL/TLS settings
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

# Create a PoolManager with the custom SSL context
http = urllib3.PoolManager(
    ssl_context=ssl_context
)

mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/Users/saip/My\ Drive/deformable-detr-soccer-analysis/data")

mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])

# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])