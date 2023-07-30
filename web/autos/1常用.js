// ==UserScript==
// @name         常用prompt
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  内置的常用prompt
// @author       FIGHTZERO
// @match        http://127.0.0.1:17860/
// @icon         https://www.google.com/s2/favicons?sz=64&domain=0.1
// @grant        none
// ==/UserScript==

app.func_menu = func = func.concat([
    {
        name: "专利改写",
        description: "对指定内容进行专利语言的改写，以辅助完成专利内容申请",
        question:
            "用专利语言改写以下段落，在修改段落时,需要确保文本的含义不发生变化,可以重新排列句子、改变表达方式。"
        ,
    },
    {
        name: "翻译",
        description: "",
        question: "翻译成中文：",
    },
    //{
    //    name: "语音输入优化",
    //    description: "处理用第三方应用语音转换的文字，精简口头禅和语气词。",
    //    question: "请用简洁明了的语言，编辑以下段落，以改善其逻辑流程，消除印刷错误，并以中文作答。请务必保持文章的原意。请从编辑以下文字开始：",
    //},
    {
        name: "摘要生成",
        description: "根据内容，提取要点并适当扩充",
        question: "使用下面提供的文本作为基础，生成一个简洁的中文摘要，突出最重要的内容，并提供对决策有用的分析。",
    },
    {
        name: "产业问题生成",
        description: "基于内容生成常见产业链知识问答",
        question: "根据以下内容，生成一个 10 个常见产业链问题的清单：",
    },
    //{
    //    name: "提问助手",
    //    description: "多角度提问，触发深度思考",
    //    question: "针对以下内容，提出疑虑和可能出现的问题，用来促进更完整的思考：",
    //},
    //{
    //    name: "评论助手",
    //    description: "",
    //    question: "针对以下内容，进行一段有评论，可以包括对作者的感谢，提出可能出现的问题等：",
    //},
    //{
    //    name: "意见回答",
    //    description: "为意见答复提供模板",
    //    question: "你是一个回复基层意见的助手，你会针对一段内容拟制回复，回复中应充分分析可能造成的后果，并从促进的单位建设的角度回答。回应以下内容：",
    //},
    {
        name: "产业问答",
        description: "",
        question: "你是一个产业链专家，你会把一个产业链问题题拆解成相关的多个子主题。请你使用中文，针对下列问题，提供相关的解答。直接输出结果，不需要额外的声明：",
    },
    //{
    //    name: "内容总结",
    //    description: "将文本内容总结为 100 字。",
    //    question: "将以下文字概括为 100 个字，使其易于阅读和理解。避免使用复杂的句子结构或技术术语。",
    //},
    //{
    //    name: "写新闻",
    //    description: "根据主题撰写新闻",
    //    question: "使用清晰、简洁、易读的语言写一篇新闻，主题为",
    //},
])
