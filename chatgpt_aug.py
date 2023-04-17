import os
import openai
import time
import sys

openai.api_key = ""

findings_path = "G:\\A\\Desktop\\CoNT Work\\ChatGPT_Aug-main\\second_for_chatgpt\\second_for_chatgpt_findings.txt"  # findings 路径 （需要重写的内容）
impression_path = "G:\\A\\Desktop\\CoNT Work\\ChatGPT_Aug-main\\second_for_chatgpt\\second_for_chatgpt_impression.txt"  # impression 路径 （需要和findings一一对应）
rewrite_path = "G:\\A\\Desktop\\CoNT Work\\ChatGPT_Aug-main\\rewrite\\rewrite_findings.txt"  # chatgpt重写好的文本保存路径
rewrite_check_path = "G:\\A\\Desktop\\CoNT Work\\ChatGPT_Aug-main\\rewrite\\rewrite_findings_check.txt"



def read_txt(txt_path):
    txtfile = open(txt_path)
    text = []
    for line in txtfile:
        text.append(line.strip("\n"))
    return text


def chatgpt_completion(model_new="gpt-3.5-turbo", prompt_new="hi", temperature_new=1, top_p_new=1, n_new=1,
                       max_tokens_new=100):
    Chat_Completion = openai.ChatCompletion.create(
        model=model_new,
        messages=[
            {"role": "user", "content": prompt_new}
        ],
        temperature=temperature_new,
        top_p=top_p_new,
        n=n_new,
        max_tokens=max_tokens_new,
        presence_penalty=0,
        frequency_penalty=0
    )
    return Chat_Completion


if __name__ == '__main__':
    findings = read_txt(findings_path)
    impression = read_txt(impression_path)

    if os.path.isfile(rewrite_path):  # 如果原先有生成的文本就先删除
        os.remove(rewrite_path)
    if os.path.isfile(rewrite_check_path):  # 如果原先有生成的文本就先删除
        os.remove(rewrite_check_path)
    for i in range(len(findings)):
        prompt = "give me 3 similar sentences like this:\n" + findings[i]  # 即输入到messages的content里的内容
        completion = chatgpt_completion(prompt_new=prompt, max_tokens_new=400)
        rewrite_finding = ""

        for line in completion.choices[0].message.content.splitlines():
            if line != "":
                sentence = line.replace(")", ".").split(". "[1], 1)[1]
                rewrite_finding = rewrite_finding + sentence + "\n"

        with open(rewrite_path, "a") as f:
            f.write(rewrite_finding)
        with open(rewrite_check_path, "a") as f:
            f.write("-----------第" + str(i + 1) + "个-----------\n")
            f.write("impression:" + impression[i]+" \n\n")
            f.write(prompt+"\n\n")
            f.write(rewrite_finding)

        print("-----------第" + str(i + 1) + "个-----------\n")
        print("impression:" + impression[i]+" \n\n")
        print(prompt+"\n\n")
        print(rewrite_finding)

        time.sleep(30)  # 国内测试10-15s的请求间隔以上可以稳定请求100次以上