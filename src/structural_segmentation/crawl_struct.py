"""Crawl structural, harmonic and thematic analysis of sonatas by Mozart, Beethoven and Haydn from https://tonic-chord.com/category/analysis, by F. Helena Marks, H.A. Harding and Eileen Stainkamph respectively.

Returns:
    _type_: _description_
"""

import os
import re
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

ROOT_URL = "https://tonic-chord.com/category/analysis"
MOV_IDX = ['first', 'second', 'third', 'fourth']
KEYS = ['struct', 'key', 'form']


def normalize_theme(theme_str):
    theme_str = theme_str.lower()
    theme_str = theme_str.replace("1st", "first")
    theme_str = theme_str.replace("2nd", "second")
    return theme_str


def normalize_form(form_str):
    form_str = form_str.lower().replace('form', '')
    form_str = form_str.split(':')[-1]
    return form_str.strip().lstrip()


def normalize_key(key_str):
    key_str = key_str.replace(' flat', 'b')
    tonic, mode = key_str.replace('.', '').split(' ')[-2:]
    mode = mode.lower()[:3]
    return f"{tonic}:{mode}"


def parse_struct(url):
    html_page = urlopen(Request(url)).read()
    soup = BeautifulSoup(html_page, 'html.parser')

    # Get Movements
    movs, forms, keys = [], [], []
    for mov in soup.find_all('span', attrs={"id": re.compile("Movement")}):
        movs.append(mov)
        form_ele = mov.find_next('p').text.replace(u'\xa0', u'')
        if ". " not in form_ele:
            eles = form_ele.split()
            form_str, key_str = ' '.join(eles[:-2]), ' '.join(eles[-2:])
        else:
            form_str, key_str = form_ele.split('. ')
        forms.append(normalize_form(form_str))
        keys.append(normalize_key(key_str))

    # Get Section
    mov_sect = {}
    for i in range(len(movs)):
        mov_name = movs[i].text
        mov_sect[mov_name] = []
        for section in movs[i].find_all_next('span'):
            if i < len(movs) - 1:
                if section.text == movs[i + 1].text:
                    break
            mov_sect[mov_name].append(section)

    mov_segment = []
    for i_mov in range(len(movs)):
        mov_name = movs[i_mov].text
        if i_mov < len(movs) - 1:
            end_tag = movs[i_mov + 1]
        else:
            end_tag = None
        sections = mov_sect[mov_name]

        segment = {}
        last_theme_content = ""
        for i, sect in enumerate(sections):
            sect_name = sect.text.replace(":", "").strip().lower()
            segment[sect_name] = []
            for ele in sections[i].find_all_next():
                if i < len(sections) - 1:
                    next_tag = sections[i + 1]
                else:
                    next_tag = end_tag
                if ele == next_tag:
                    break

                bars = ele.find('strong') or ele.find('b')
                if not bars and 'Bars' in ele.text:
                    bars = ele

                if bars and 'Bars' in bars.text:

                    if bars.text == last_theme_content:
                        continue
                    else:
                        last_theme_content = bars.text

                    bar_str = bars.text.replace("Bars", "")
                    bar_idx = bar_str.replace(":", "").strip().split("-")
                    if len(bar_idx) < 2:
                        bar_idx.append(bar_idx[0])
                    entry = {"start": int(bar_idx[0]) if bar_idx[0].isdigit() else bar_idx[0],
                             "end": int(bar_idx[1]) if bar_idx[1].isdigit() else bar_idx[1]}

                    theme = ""
                    for sib in bars.next_siblings:
                        if sib.name == 'em':
                            theme = sib.text.replace(u'\xa0', u' ').strip()
                            break
                    entry['theme'] = normalize_theme(theme)

                    segment[sect_name].append(entry)
                else:
                    continue

        mov_segment.append({'struct': segment,
                            'key': keys[i_mov],
                            'form': forms[i_mov],
                            'title': mov_name})
    return mov_segment

# def parse_form(url):
#     req = Request(url)
#     html_page = urlopen(req).read()
#     soup = BeautifulSoup(html_page, 'html.parser')

#     # Get Movements
#     mov_segment = {}
#     for mov in soup.find_all('span', attrs={"id": re.compile("Movement")}):
#         mov_name = mov.text
#         form_ele = mov.find_next('p').text.replace(u'\xa0', u'')

#         if ". " not in form_ele:
#             eles = form_ele.split()
#             form_str, key_str = ' '.join(eles[:-2]), ' '.join(eles[-2:])
#         else:
#             form_str, key_str = form_ele.split('. ')
#         mov_segment[mov_name] = {"form": normalize_form(form_str),
#                                  "key": normalize_key(key_str)}
#     return mov_segment


def main():

    os.makedirs(RAW_DIR, exist_ok=True)

    for composer in COMPOSERS:

        raw_dir = os.path.join(RAW_DIR, composer)
        os.makedirs(raw_dir, exist_ok=True)

        prefix = f"{composer}-piano-sonata"
        if composer == "haydn":
            to_remove = f"{prefix}-"
        else:
            to_remove = f"{prefix}-in-"

        urls = [f"{ROOT_URL}/{prefix}s"]
        i_page = 0
        while urls:
            i_page += 1

            url = urls.pop()
            req = Request(url)
            html_page = urlopen(req).read()
            soup = BeautifulSoup(html_page, 'html.parser')

            # Fetch url to next page
            next_page = soup.find("a", attrs={"class": "next page-numbers"})
            if next_page:
                urls.append(next_page['href'])

            all_urls = []
            for entry in soup.find_all("a", attrs={"href": re.compile(f"{prefix}-")}):
                all_urls.append(entry['href'])
            all_urls = list(set(all_urls))

            # Iterate through all the analysis on this page
            for analysis_url in tqdm(all_urls, desc=f"{composer} page {i_page}"):

                # Get structural analysis for one sonata
                mov_struct = parse_struct(analysis_url)

                base_name = analysis_url.split("/")[-2].replace(to_remove, "")
                base_name = base_name.replace('-analysis', '')

                # Get sonata number from url
                if composer == "haydn":
                    matched_xvi = re.search("xvi\d+", base_name)
                    if not matched_xvi:
                        print(f"Sonata No. not found in {base_name}")
                        continue
                    xvi = matched_xvi.group().replace("xvi", "")
                    sonata_no = HAYDN_XVI_NO_DICT.get(xvi, None)
                    if not sonata_no:
                        print(f"{base_name} is not in the dataset.")
                        continue
                else:
                    matched = re.search("no-\d+", base_name)
                    if not matched:
                        print(f"Sonata No. not found in {base_name}")
                        continue
                    sonata_no = matched.group().replace("no-", "")

                # Split structural analysis by movement
                try:
                    for i_mov, item in enumerate(mov_struct):

                        # Sanity check
                        assert MOV_IDX[i_mov] in item['title'].lower()
                        mov_base_name = f"sonata{int(sonata_no):02d}-{i_mov + 1}.json"
                        mov_fname = os.path.join(raw_dir, mov_base_name)

                        with open(mov_fname, "w") as f:
                            json.dump({key: item[key] for key in KEYS}, f)
                except:
                    print(f"Manually check {composer}/{base_name}.")
                    continue


if __name__ == "__main__":
    import pandas as pd

    # COMPOSERS = ["mozart", "beethoven", 'haydn']
    COMPOSERS = ['haydn']
    RAW_DIR = "./raw"

    # Match Hoboken to Sonata no.
    HAYDN_INFO_FILE = "../../sonata-dataset/info/haydn.csv"

    df = pd.read_csv(HAYDN_INFO_FILE)
    HAYDN_XVI_NO_DICT = {}
    XVI_PATTERN = r"XVI:\d+"
    NO_PATTERN = r"No. \d+"
    for title in set(df['title']):
        xvi_matched = re.search(XVI_PATTERN, title)
        no_matched = re.search(NO_PATTERN, title)
        if xvi_matched is None or no_matched is None:
            continue
        xvi = xvi_matched.group().replace("XVI:", "")
        no = no_matched.group().replace("No. ", "")
        HAYDN_XVI_NO_DICT[xvi] = no

    main()
