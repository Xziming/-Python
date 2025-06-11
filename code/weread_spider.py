from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import time
import csv
import json
import re
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeReadScraper:
    def __init__(self, driver_path=None, headless=True, get_detail=True):
        """
        初始化爬虫

        Args:
            driver_path: ChromeDriver路径，如果为None则使用系统PATH中的chromedriver
            headless: 是否使用无头模式
            get_detail: 是否获取书籍详情页信息
        """
        self.driver_path = driver_path
        self.headless = headless
        self.get_detail = get_detail
        self.driver = None

    def setup_driver(self):
        """设置Chrome浏览器驱动"""
        try:
            options = webdriver.ChromeOptions()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-proxy-server')
            options.add_argument('--proxy-server=""')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)

            # 添加用户代理
            options.add_argument(
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            if self.driver_path:
                service = Service(self.driver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)

            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return True

        except Exception as e:
            logger.error(f"设置浏览器驱动失败: {e}")
            return False

    def get_page_content(self, url, scroll_times=10, wait_time=1):
        """
        获取页面内容

        Args:
            url: 目标URL
            scroll_times: 滚动次数
            wait_time: 每次滚动后等待时间
        """
        try:
            self.driver.get(url)
            logger.info(f"正在访问: {url}")

            # 等待页面加载
            time.sleep(2)

            # 滚动页面加载更多内容
            for i in range(scroll_times):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(wait_time)
                logger.info(f"完成第 {i + 1} 次滚动")

            # 等待书籍列表加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'wr_bookList_item'))
            )

            return self.driver.page_source

        except TimeoutException:
            logger.warning(f"页面加载超时: {url}")
            return self.driver.page_source if self.driver else None
        except Exception as e:
            logger.error(f"获取页面内容失败: {e}")
            return None

    def get_book_detail_content(self, url):
        """获取书籍详情页内容"""
        try:
            self.driver.get(url)
            logger.info(f"正在访问书籍详情页: {url}")

            # 等待页面加载
            time.sleep(1)

            return self.driver.page_source

        except Exception as e:
            logger.error(f"获取书籍详情页失败: {e}")
            return None

    def parse_books(self, page_content):
        """解析书籍信息"""
        if not page_content:
            return []

        soup = BeautifulSoup(page_content, 'html.parser')
        books = []
        book_items = soup.find_all('li', class_='wr_bookList_item')

        logger.info(f"找到 {len(book_items)} 个书籍项目")

        for i, book_item in enumerate(book_items, 1):
            try:
                book_data = self._extract_basic_book_data(book_item)
                if book_data:
                    # 如果需要获取详情页信息
                    if self.get_detail and book_data.get('url'):
                        detail_data = self._get_book_detail_data(book_data['url'])
                        if detail_data:
                            book_data.update(detail_data)

                    books.append(book_data)
                    logger.info(f"成功提取第 {i} 本书: {book_data['title']}")
                else:
                    logger.warning(f"跳过第 {i} 本书（缺少关键信息）")

            except Exception as e:
                logger.error(f"解析第 {i} 本书失败: {e}")
                continue

        return books

    def _extract_basic_book_data(self, book_item):
        """提取单本书籍基本数据"""
        # 获取基本信息
        title_tag = book_item.find('p', class_='wr_bookList_item_title')
        author_tag = book_item.find('p', class_='wr_bookList_item_author')

        if not title_tag or not author_tag:
            return None

        title = title_tag.text.strip()
        author = author_tag.text.strip()

        # 获取书籍链接
        book_link = book_item.find('a', class_='wr_bookList_item_link')
        book_url = ''
        if book_link and book_link.get('href'):
            book_url = f"https://weread.qq.com{book_link.get('href')}"

        # 获取阅读数量
        reading_count = self._extract_reading_count(book_item)

        # 获取推荐信息
        recommendation_tag = book_item.find('span', class_='wr_bookList_item_reading_percent')
        recommendation = recommendation_tag.text.strip() if recommendation_tag else '无推荐'

        # 获取评价
        evaluation = self._extract_evaluation(book_item)

        return {
            'title': title,
            'author': author,
            'reading_count': reading_count,
            'recommendation': recommendation,
            'evaluation': evaluation,
            'url': book_url
        }

    def _get_book_detail_data(self, book_url):
        """获取书籍详情页数据"""
        try:
            detail_content = self.get_book_detail_content(book_url)
            if not detail_content:
                return {}

            soup = BeautifulSoup(detail_content, 'html.parser')
            detail_data = {}

            # 提取ISBN
            isbn = self._extract_isbn(soup)
            if isbn:
                detail_data['isbn'] = isbn

            # 提取出版信息
            pub_info = self._extract_publication_info(soup)
            detail_data.update(pub_info)

            # 提取点评信息
            rating_info = self._extract_rating_info(soup)
            detail_data.update(rating_info)

            # 提取总阅读人数
            total_readers = self._extract_total_readers(soup)
            if total_readers:
                detail_data['total_readers'] = total_readers

            # 提取简介
            intro = self._extract_introduction(soup)
            if intro:
                detail_data['introduction'] = intro

            time.sleep(1)  # 延迟避免请求过频
            return detail_data

        except Exception as e:
            logger.error(f"获取书籍详情失败: {e}")
            return {}

    def _extract_isbn(self, soup):
        """从JSON-LD脚本中提取ISBN"""
        try:
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'isbn' in data:
                        return data['isbn']
                except json.JSONDecodeError:
                    continue
            return None
        except Exception:
            return None

    def _extract_publication_info(self, soup):
        """提取出版信息"""
        pub_info = {
            'publisher': '',
            'publish_time': '',
            'word_count': '',
            'detailed_category': '',
            'category': ''

        }

        try:
            # 查找简介对话框内容
            intro_content = soup.find('div', class_='introDialog_content')
            if intro_content:
                pub_lines = intro_content.find_all('div', class_='introDialog_content_pub_line')
                for line in pub_lines:
                    title_span = line.find('span', class_='introDialog_content_pub_title')
                    if title_span:
                        title = title_span.text.strip()
                        value_span = title_span.find_next_sibling('span')
                        if value_span:
                            value = value_span.text.strip()
                            if title == '出版社':
                                pub_info['publisher'] = value
                            elif title == '出版时间':
                                pub_info['publish_time'] = value
                            elif title == '字数':
                                pub_info['word_count'] = value
                            elif title == '分类':
                                # 拆分分类
                                if '-' in value:
                                    cat, detail_cat = value.split('-', 1)
                                    pub_info['category'] = cat.strip()
                                    pub_info['detailed_category'] = detail_cat.strip()
                                else:
                                    pub_info['detailed_category'] = value.strip()
                                    pub_info['category'] = ''

        except Exception as e:
            logger.error(f"提取出版信息失败: {e}")

        return pub_info

    def _extract_rating_info(self, soup):
        """提取点评信息"""
        rating_info = {
            'rating_count': 0,
            'recommend_count': 0,
            'general_count': 0,
            'bad_count': 0
        }

        try:
            # 提取点评人数
            rating_count_elem = soup.find('div', class_='book_rating_item_detail_count')
            if rating_count_elem:
                count_text = rating_count_elem.find('span')
                if count_text:
                    count_str = count_text.text.strip()
                    rating_info['rating_count'] = self._parse_number(count_str)

            # 提取推荐/一般/不行的数量
            rating_buttons = soup.find('div', class_='book_ratings_buttons')
            if rating_buttons:
                buttons = rating_buttons.find_all('div', class_='book_ratings_button')
                for button in buttons:
                    span = button.find('span')
                    if span:
                        text = span.text.strip()
                        if '推荐' in text:
                            rating_info['recommend_count'] = self._extract_number_from_text(text)
                        elif '一般' in text:
                            rating_info['general_count'] = self._extract_number_from_text(text)
                        elif '不行' in text:
                            rating_info['bad_count'] = self._extract_number_from_text(text)

        except Exception as e:
            logger.error(f"提取点评信息失败: {e}")

        return rating_info

    def _extract_total_readers(self, soup):
        """提取总阅读人数"""
        try:
            # 查找总阅读人数元素
            reader_elem = soup.find('div', class_='horizontalReaderCoverPage_stats_item_data')
            if reader_elem:
                span = reader_elem.find('span')
                if span:
                    text = span.text.strip()
                    return self._parse_number(text)
            return 0
        except Exception:
            return 0

    def _extract_introduction(self, soup):
        """提取简介"""
        try:
            intro_elem = soup.find('p', class_='introDialog_content_intro_para')
            if intro_elem:
                return intro_elem.text.strip()
            return ''
        except Exception:
            return ''

    def _parse_number(self, text):
        """解析数字，支持万、千等单位"""
        try:
            # 移除非数字和小数点字符，保留万、千等单位
            clean_text = re.sub(r'[^\d\.\万千]', '', text)

            if '万' in clean_text:
                number_part = clean_text.replace('万', '')
                return int(float(number_part) * 10000)
            elif '千' in clean_text:
                number_part = clean_text.replace('千', '')
                return int(float(number_part) * 1000)
            else:
                # 提取纯数字
                numbers = re.findall(r'\d+\.?\d*', clean_text)
                if numbers:
                    return int(float(numbers[0]))
            return 0
        except (ValueError, TypeError):
            return 0

    def _extract_number_from_text(self, text):
        """从文本中提取数字（如：推荐(243316)）"""
        try:
            numbers = re.findall(r'\((\d+)\)', text)
            if numbers:
                return int(numbers[0])
            return 0
        except (ValueError, TypeError):
            return 0

    def _extract_reading_count(self, book_item):
        """提取阅读数量"""
        reading_count_text_tag = book_item.find('span', class_='wr_bookList_item_readingCountText')
        reading_count_number_tag = book_item.find('span', class_='wr_bookList_item_reading_number')

        if not reading_count_text_tag or not reading_count_number_tag:
            return 0

        try:
            reading_count_text = reading_count_text_tag.text.strip()
            reading_count_number = reading_count_number_tag.text.strip()

            if '万' in reading_count_text:
                return int(float(reading_count_number) * 10000)
            else:
                return int(reading_count_number)
        except (ValueError, TypeError):
            return 0

    def _extract_evaluation(self, book_item):
        """提取评价信息"""
        rating_img = book_item.find('img', class_='book_rating_item_label_number_image')
        if not rating_img:
            return '无评价'

        src = rating_img.get('src', '')
        rating_map = {
            'newRatings_900': '神作',
            'newRatings_870': '神作/潜力',
            'newRatings_850': '好评如潮',
            'newRatings_800': '脍炙人口',
            'newRatings_700': '值得一读'
        }

        for key, value in rating_map.items():
            if key in src:
                return value

        return '无评价'

    def save_to_csv(self, books, filename='weread_books.csv'):
        """保存数据到CSV文件"""
        if not books:
            logger.warning("没有数据需要保存")
            return

        try:
            fieldnames = [
                'title', 'author', 'isbn', 'reading_count', 'total_readers',
                'recommendation', 'evaluation', 'category', 'detailed_category',
                'publisher', 'publish_time', 'word_count',
                'rating_count', 'recommend_count', 'general_count', 'bad_count',
                'introduction', 'url'
            ]

            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                for book in books:
                    # 确保所有字段都存在，缺失的用默认值填充
                    row = {}
                    for field in fieldnames:
                        row[field] = book.get(field, '')
                    writer.writerow(row)

            logger.info(f"成功保存 {len(books)} 本书籍信息到: {filename}")

        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")

    def scrape_targeturl(self, url, output_file='weread_books_detailed.csv'):
        """爬取所有分类的书籍"""
        if not self.setup_driver():
            logger.error("无法初始化浏览器驱动")
            return

        all_books = []

        try:
            for url in url:
                logger.info(f"开始爬取: {url}")

                page_content = self.get_page_content(url)
                if page_content:
                    books = self.parse_books(page_content)
                    all_books.extend(books)
                    logger.info(f"链接 '{url}' 成功爬取 {len(books)} 本书")
                else:
                    logger.warning(f"链接 '{url}' 爬取失败")

                # 添加延迟避免请求过频
                time.sleep(3)

        finally:
            if self.driver:
                self.driver.quit()
                logger.info("浏览器驱动已关闭")

        self.save_to_csv(all_books, output_file)
        return all_books

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()


def main():
    """主函数"""
    url = [
        'https://weread.qq.com/web/category/100001',
        'https://weread.qq.com/web/category/100002',
        'https://weread.qq.com/web/category/100003',
        'https://weread.qq.com/web/category/100004',
        'https://weread.qq.com/web/category/100005',
        'https://weread.qq.com/web/category/100006',
        'https://weread.qq.com/web/category/100007',
        'https://weread.qq.com/web/category/100008',
        'https://weread.qq.com/web/category/100009',
        'https://weread.qq.com/web/category/100010',
        'https://weread.qq.com/web/category/100011',
        'https://weread.qq.com/web/category/100012',
        'https://weread.qq.com/web/category/100014',
        'https://weread.qq.com/web/category/100015',
        'https://weread.qq.com/web/category/200001',
        'https://weread.qq.com/web/category/200002',
        'https://weread.qq.com/web/category/200003',
        'https://weread.qq.com/web/category/200004',
        'https://weread.qq.com/web/category/200005',
        'https://weread.qq.com/web/category/200006',
        'https://weread.qq.com/web/category/200007',
        'https://weread.qq.com/web/category/200008',
        'https://weread.qq.com/web/category/200009',
        'https://weread.qq.com/web/category/300001',
        'https://weread.qq.com/web/category/300002',
        'https://weread.qq.com/web/category/300003',
        'https://weread.qq.com/web/category/300004',
        'https://weread.qq.com/web/category/300005',
        'https://weread.qq.com/web/category/300006',
        'https://weread.qq.com/web/category/300007',
        'https://weread.qq.com/web/category/300008',
        'https://weread.qq.com/web/category/300009',
        'https://weread.qq.com/web/category/300010',
        'https://weread.qq.com/web/category/300011',
        'https://weread.qq.com/web/category/300012',
        'https://weread.qq.com/web/category/300013',
        'https://weread.qq.com/web/category/300014',
        'https://weread.qq.com/web/category/400001',
        'https://weread.qq.com/web/category/400002',
        'https://weread.qq.com/web/category/400003',
        'https://weread.qq.com/web/category/400004',
        'https://weread.qq.com/web/category/400005',
        'https://weread.qq.com/web/category/400006',
        'https://weread.qq.com/web/category/400007',
        'https://weread.qq.com/web/category/400008',
        'https://weread.qq.com/web/category/400009',
        'https://weread.qq.com/web/category/400010',
        'https://weread.qq.com/web/category/400011',
        'https://weread.qq.com/web/category/400012',
        'https://weread.qq.com/web/category/400013',
        'https://weread.qq.com/web/category/400014',
        'https://weread.qq.com/web/category/500001',
        'https://weread.qq.com/web/category/500002',
        'https://weread.qq.com/web/category/500003',
        'https://weread.qq.com/web/category/500004',
        'https://weread.qq.com/web/category/500005',
        'https://weread.qq.com/web/category/500006',
        'https://weread.qq.com/web/category/500007',
        'https://weread.qq.com/web/category/500008',
        'https://weread.qq.com/web/category/500009',
        'https://weread.qq.com/web/category/500010',
        'https://weread.qq.com/web/category/600001',
        'https://weread.qq.com/web/category/600002',
        'https://weread.qq.com/web/category/600003',
        'https://weread.qq.com/web/category/600004',
        'https://weread.qq.com/web/category/600005',
        'https://weread.qq.com/web/category/600006',
        'https://weread.qq.com/web/category/600007',
        'https://weread.qq.com/web/category/600008',
        'https://weread.qq.com/web/category/600009',
        'https://weread.qq.com/web/category/600010',
        'https://weread.qq.com/web/category/700001',
        'https://weread.qq.com/web/category/700002',
        'https://weread.qq.com/web/category/700003',
        'https://weread.qq.com/web/category/700004',
        'https://weread.qq.com/web/category/700005',
        'https://weread.qq.com/web/category/700006',
        'https://weread.qq.com/web/category/700007',
        'https://weread.qq.com/web/category/800001',
        'https://weread.qq.com/web/category/800002',
        'https://weread.qq.com/web/category/800003',
        'https://weread.qq.com/web/category/800004',
        'https://weread.qq.com/web/category/800005',
        'https://weread.qq.com/web/category/800006',
        'https://weread.qq.com/web/category/800007',
        'https://weread.qq.com/web/category/900001',
        'https://weread.qq.com/web/category/900002',
        'https://weread.qq.com/web/category/900003',
        'https://weread.qq.com/web/category/1100001',
        'https://weread.qq.com/web/category/1100002',
        'https://weread.qq.com/web/category/1100003',
        'https://weread.qq.com/web/category/1100004',
        'https://weread.qq.com/web/category/1200001',
        'https://weread.qq.com/web/category/1200002',
        'https://weread.qq.com/web/category/1500001',
        'https://weread.qq.com/web/category/1500002',
        'https://weread.qq.com/web/category/1500003',
        'https://weread.qq.com/web/category/1500004',
        'https://weread.qq.com/web/category/1500005',
    ]

    scraper = WeReadScraper(get_detail=True)

    # 使用上下文管理器确保资源正确释放
    with scraper:
        books = scraper.scrape_targeturl(url, '../data/weread_books_detailed.csv')
        logger.info(f"爬取完成，总共获取 {len(books)} 本书籍信息")


if __name__ == '__main__':
    main()

