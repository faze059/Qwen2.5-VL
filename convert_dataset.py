import json, os; src="hf_dataset/train.jsonl"; out="data/train_72b_lora.jsonl"; os.makedirs(os.path.dirname(out), exist_ok=True); dir_files=set(os.listdir("data/protocols_max1000px")); cnt=0; fout=open(out,"w");
for line in open(src):
	item=json.loads(line); img_path=item.get("images") or item.get("image");
	if not img_path: continue;
	fn=os.path.basename(img_path);
	new_img=os.path.join("data","protocols_max1000px",fn);
	if not os.path.isfile(new_img):
		match=next((f2 for f2 in dir_files if f2.lower()==fn.lower()), None);
		if match: new_img=os.path.join("data","protocols_max1000px",match);
		else: continue;
	msgs=item.get("messages") or item.get("conversations");
	if not msgs: continue;
	convs=[];
	for m in msgs:
		role=m.get("role") or m.get("from"); content=m.get("content") or m.get("value"); frm="human" if role in ["user","human"] else ("gpt" if role in ["assistant","gpt"] else role); convs.append({"from":frm,"value":content});
	json.dump({"image":new_img, "conversations":convs}, fout); fout.write("\n"); cnt+=1;
print("Converted", cnt); fout.close()
