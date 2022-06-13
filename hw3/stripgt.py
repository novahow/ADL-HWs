import jsonlines
import json
with jsonlines.open('../data/public.jsonl', 'r') as f:
    with open(f'../data/strip.jsonl', "w", encoding="utf8") as fp:
        with open(f'../data/gt.jsonl', "w", encoding="utf8") as gt:
            for article in f:
                title = article['title'] if 'title' in article.keys() else ''
                aid = article['id']
                context = article['maintext']    
                json.dump({"id":aid, "maintext":context}, fp, ensure_ascii = False)
                fp.write("\n")
                json.dump({"id":aid, "title":title}, gt, ensure_ascii = False)
                gt.write("\n")
        