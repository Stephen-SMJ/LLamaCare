import json


# json, k,v
def process_kv(path_to_data,save_name):
    new_data = []
    with open(path_to_data, 'r') as f:
        data = json.load(f)
        for _, val in data.items():
            new_data.append({
                'text':(
                    "instruction:{ins}\n"
                    "output:{out}"
                ).format(ins=val['prompt_question'],
                         out=val['prompt_answer']),
                'final_decision': val['final_decision']

            })
        f.close()
    with open(save_name, 'w', encoding='utf-8') as f:
        print(json.dumps(new_data, ensure_ascii=False, indent=4), file=f)
        f.close()
    # with open(save_name, 'w', encoding='utf-8') as f:
    #     for line in new_data:
    #         print(json.dumps(line, ensure_ascii=False, indent=4), file=f)
    #     f.close()

# josn, list
def process_list(path_to_data,save_name):
    new_data = []
    with open(path_to_data, 'r') as f:
        data = json.load(f)
        for i in data:
            new_data.append({
                'text': (
                    "instruction:{ins}\n"
                    "output:{out}"
                ).format(ins=i['prompt_question'],
                         out=i['prompt_answer']),
                'final_decision': i['final_decision']
            })
        f.close()

    with open(save_name, 'w', encoding='utf-8') as f:
        for line in new_data:
            print(json.dumps(line, ensure_ascii=False, indent=4))
        f.close()

if __name__ == '__main__':
    process_kv("pqal_pmt.json","processed_data/pqal_pmt_process.json")
