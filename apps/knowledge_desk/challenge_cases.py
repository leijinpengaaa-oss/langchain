"""Frozen, self-authored paraphrase set; development/validation are not independent data."""

GROUPS = [
    (
        "运行环境",
        [
            "给星舟配一台机器，最少得有多少算力和内存？",
            "What runtime and database versions does Xingzhou require?",
            "我要给星舟开虚拟机，配置门槛有哪些？",
        ],
    ),
    (
        "服务端口与健康检查",
        [
            "负载均衡想探测星舟还活着，该访问哪个地址？",
            "Xingzhou readiness probe URL and timeout?",
            "监控要确认星舟在线，探活接口和正常响应怎么配？",
        ],
    ),
    (
        "启动失败排查",
        [
            "星舟起不来，排障从哪儿开始看？",
            "Where should I look when Xingzhou cannot boot?",
            "星舟启动一直报错，定位问题的先后次序是什么？",
        ],
    ),
    (
        "优雅停机",
        [
            "星舟进程准备退出，手头没干完的活咋办？",
            "How does Xingzhou drain in-flight work on SIGTERM?",
            "星舟下线时会给正在跑的任务留多长收尾时间？",
        ],
    ),
    (
        "数据备份与恢复",
        [
            "星舟的库坏了，找哪天的副本，怎样安全救回来？",
            "Xingzhou backup schedule, retention, and restore drill?",
            "星舟的历史数据副本几点留、能留多久？",
        ],
    ),
    (
        "发布与回滚",
        [
            "星舟上线后大量报错，什么条件下退回老版本？",
            "When should a Xingzhou release be rolled back?",
            "星舟升级完错误变多了，降回旧版前要留哪些东西？",
        ],
    ),
    (
        "身份认证与密钥轮换",
        [
            "星舟换访问令牌时，旧的还能用多久？",
            "How to rotate Xingzhou API credentials without downtime?",
            "调用星舟接口的凭据放网址里行吗，隔多久换一次？",
        ],
    ),
    (
        "请求限流",
        [
            "星舟说我请求太频繁了，缓多久再发？",
            "What should a Xingzhou client do on HTTP429?",
            "星舟一个租户一分钟能调几回，能同时跑几件事？",
        ],
    ),
    (
        "幂等请求",
        [
            "星舟建单点了两次，怎样还拿到同一个编号？",
            "Which idempotency key is needed for Xingzhou ticket creation?",
            "星舟建单同一个请求标识换了内容，服务怎么处理？",
        ],
    ),
    (
        "错误码处理",
        [
            "星舟说没权限或者凭据失效，能无限重发吗？",
            "Which Xingzhou status codes should never be retried?",
            "星舟返回５０３，重发上限和等待策略是啥？",
        ],
    ),
    (
        "文件上传限制",
        [
            "星舟不认识我扫出来的文档，怎么让它读到字？",
            "Can Xingzhou ingest scanned PDFs directly?",
            "星舟收资料时，大小、页数和字符编码有啥门槛？",
        ],
    ),
    (
        "会话与权限",
        [
            "星舟两个人能互相看聊天和资料吗？",
            "How does Xingzhou isolate conversations and concurrent edits?",
            "星舟两条消息同时改一个聊天线程会怎样？",
        ],
    ),
    (
        "工单必填信息",
        [
            "给星舟提问题单，要把哪些故障细节交代清楚？",
            "What must a Xingzhou incident report contain?",
            "写星舟问题单时，哪些敏感内容绝对不能塞进去？",
        ],
    ),
    (
        "工单优先级与响应时间",
        [
            "星舟线上彻底挂了，值班的人多久先回复我？",
            "Xingzhou SLA for high, normal and low priority incidents?",
            "星舟只是咨询优化，最迟几个工作日收到第一次回应？",
        ],
    ),
    (
        "工单分类",
        [
            "星舟连不上库，这张问题单该分给哪类？",
            "Which category handles Xingzhou launch and release failures?",
            "星舟接口请求太多被拒，该归账号还是接口类？",
        ],
    ),
    (
        "审核与取消",
        [
            "星舟还没点确认就断电，问题单草稿回来还能审吗？",
            "Does rejecting a Xingzhou draft create a ticket?",
            "星舟机器重开后，没审完的草稿还能接着处理吗？",
        ],
    ),
    (
        "工单状态流转",
        [
            "星舟问题已经处理好了，单子接下来怎么标记？",
            "What is the Xingzhou ticket lifecycle?",
            "星舟的问题单从收到到办完有哪些状态？",
        ],
    ),
    (
        "重试与重复提交",
        [
            "星舟已经存了单但我没收到回包，再跑会不会多一张？",
            "How to recover a Xingzhou ticket operation after a lost response?",
            "星舟建单后断网，用老请求编号再发会怎样？",
        ],
    ),
]

CHALLENGE_CASES = [
    {
        "id": f"paraphrase-{group:02d}-{index}",
        "question": question,
        "headings": [heading],
        "kind": "positive",
        "split": "validation" if index == 2 else "development",
        "tag": "english" if index == 1 else "paraphrase",
    }
    for group, (heading, questions) in enumerate(GROUPS)
    for index, question in enumerate(questions)
]
CHALLENGE_CASES += [
    {
        "id": f"unrelated-{index:02d}",
        "question": question,
        "headings": [],
        "kind": "out_of_scope",
        "split": "validation" if index % 3 == 2 else "development",
        "tag": "unrelated",
    }
    for index, question in enumerate(
        [
            "怎么煮意大利面？",
            "Which camera should I buy for travel?",
            "猫咪为什么喜欢纸箱？",
            "How far is Mars from the sun?",
            "给我写一首秋天的诗",
            "明天上海是否下雨？",
            "What is the capital of Canada?",
            "英国国王是谁？",
            "南极企鹅有哪些种类？",
            "烤蛋糕要放多少糖？",
            "电影盗梦空间讲了什么？",
            "太阳的温度有多高？",
        ]
    )
]
CHALLENGE_CASES += [
    {
        "id": f"missing-fact-{index:02d}",
        "question": question,
        "headings": [heading],
        "kind": "unsupported_fact",
        "split": "validation" if index % 2 else "development",
        "tag": "near_negative",
    }
    for index, (question, heading) in enumerate(
        [
            ("星舟工单超过响应时限，每小时赔偿多少钱？", "工单优先级与响应时间"),
            ("星舟数据库备份的加密密码是什么？", "数据备份与恢复"),
            ("星舟API密钥当前的真实值是什么？", "身份认证与密钥轮换"),
            ("星舟每次服务重启的计费价格是多少？", "优雅停机"),
            ("星舟生产机器的公网IP地址是什么？", "运行环境"),
            ("星舟上传文档后用于训练的保留期限是多少？", "文件上传限制"),
        ]
    )
]

ANSWER_CASES = [
    {
        "question": "星舟下线时会给正在跑的任务留多长收尾时间？",
        "keywords": ["45"],
        "refusal": False,
    },
    {
        "question": "星舟一个租户一分钟能调几回，能同时跑几件事？",
        "keywords": ["60", "4"],
        "refusal": False,
    },
    {
        "question": "Xingzhou backup schedule and retention? 用中文回答。",
        "keywords": ["02:30", "14"],
        "refusal": False,
    },
    {
        "question": "星舟工单超过响应时限，每小时赔偿多少钱？",
        "keywords": [],
        "refusal": True,
    },
    {"question": "星舟数据库备份的加密密码是什么？", "keywords": [], "refusal": True},
    {"question": "怎么煮意大利面？", "keywords": [], "refusal": True},
]
