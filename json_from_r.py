import json


def get_json(res_li):
  results_lis = []
  i = 0
  new_dict = {}
  while (i < len(res_li)):
    if res_li[i].startswith('Title:'):
      new_dict = {}
      new_dict["title"] = res_li[i].split(':', 1)[1].strip()
    elif res_li[i].startswith('URL:'):
      new_dict["url"] = res_li[i].split(':', 1)[1].strip()
    # elif res_li[i].startswith('ID:'):
    #   new_dict["id"] = res_li[i].split(':', 1)[1].strip()
    # elif res_li[i].startswith('Score:'):
    #   new_dict["score"] = res_li[i].split(':', 1)[1].strip()
    # elif res_li[i].startswith('Published Date:'):
    #   new_dict["published_date"] = res_li[i].split(':', 1)[1].strip()
    # elif res_li[i].startswith('Author:'):
    #   new_dict["author"] = res_li[i].split(':', 1)[1].strip()
    elif res_li[i].startswith('Text:'):
      new_dict["text"] = res_li[i].split(':', 1)[1].strip()
      i+=1
      while not res_li[i].startswith('Highlights:'):
        new_dict['text'] += res_li[i]
        i+=1
    # elif res_li[i].startswith('Highlights:'):
    #   new_dict["highlights"] = res_li[i].split(':', 1)[1].strip()
    # elif res_li[i].startswith('Highlights Scores:'):
    #   new_dict["highlights_scores"] = res_li[i].split(':', 1)[1].strip
    elif res_li[i].startswith('Summary:'):
      new_dict["summary"] = res_li[i].split(':', 1)[1].strip()
      results_lis.append(new_dict)
      i += 2
    i += 1
  return results_lis


def json_from_r(res_li):
  res_js = get_json(res_li)
  json_string = json.dumps(res_js)
  parsed_data = json.loads(json_string)
  return parsed_data
