from icrawler.builtin import GoogleImageCrawler
import os
from multiprocessing import Pool, cpu_count

def crawl_images(args):
    big, small, img_path = args
    output_dir = os.path.abspath(os.path.join(img_path, big, small))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    keyword = f"{small}"
    google_crawler = GoogleImageCrawler(storage={'root_dir': output_dir})
    google_crawler.crawl(keyword=keyword, max_num=200, max_idle_time=60)

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), 'image')
    tasks = []
    for big in os.listdir(img_path):
        big_path = os.path.join(img_path, big)
        if not os.path.isdir(big_path):
            continue
        for small in os.listdir(big_path):
            small_path = os.path.join(big_path, small)
            if not os.path.isdir(small_path):
                continue
            tasks.append((big, small, img_path))

    with Pool(processes=min(cpu_count(), 8)) as pool:
        pool.map(crawl_images, tasks)
