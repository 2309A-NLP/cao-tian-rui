import json,urllib.request,urllib.parse
TOKEN="ragflow-reparse-tmp"; CHAT="40d4a4e468ab11f1868a810dbccbbded"
BASE="http://localhost:9380/api/v1/chats/"+CHAT
H={"Authorization":"Bearer "+TOKEN,"Content-Type":"application/json"}
def post(path,body):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers=H,method="POST")
    return json.load(urllib.request.urlopen(req,timeout=120))
qs=[
("Q1/A","根据文本信息，该静电除尘器的发明人是？","吉特勒"),
("Q2/C","该静电除尘器的管状入口具有怎样的结构特征？","80至95%"),
("Q3/A","在第7页图片中，部件4相对于部件5在图中的位置关系是？","左"),
("Q4/A","在第7页图片中，尺寸X1、X2、X3分别代表什么部件之间的间隔距离？","6"),
("Q5/C","根据第7页图示，气流方向(7)首先经过哪个部件？紧接着经过哪个部件？","6″"),
("Q6/B","根据第7页图示，已知外壳直径D，h1和h2的尺寸可以用来计算/确定什么？","位置"),
]
for tag,q,key in qs:
    # 每题新建会话
    s=post("/sessions",{"name":tag})
    sid=s["data"]["id"]
    r=post("/completions",{"question":q,"session_id":sid,"stream":False})
    ans=r.get("data",{}).get("answer","") if r.get("code")==0 else "ERR:"+str(r.get("message"))[:80]
    hit="✅" if key in ans else "❓"
    print(f"\n##### {tag} {hit}")
    print(ans[:300])
