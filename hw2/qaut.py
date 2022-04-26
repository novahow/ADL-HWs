import torch
import random
def evaluate(data, output, tokenizer):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    # print('quat 7', data)
    answer = ''
    max_prob = float('-inf')
    num_of_windows = len(output)
    
    for k, e in enumerate(output):
        # Obtain answer by choosing the most probable start position / end position
        
        lsp = -1
        fsp = -1
        for i in range(len(data['input_ids'][k]) - 1, -1, -1):
            t = data['input_ids'][k][i]
            if t == tokenizer.sep_token_id:
                if lsp == -1:
                    lsp = i
                else:
                    fsp = i
                    break
        
        if random.random() < 0.07:
            pass
            # print(fsp, lsp)
        beam = 4
        # Probability of answer is calculated as sum of start_prob and end_prob
        start_prob, start_index = torch.topk(e.start_logits, beam,  dim=-1)
        end_prob, end_index = torch.topk(e.end_logits, beam, dim=-1)
        start_prob = start_prob.squeeze(0)
        start_index = start_index.squeeze(0)
        end_prob = end_prob.squeeze(0)
        end_index = end_index.squeeze(0)
        # print('22', start_prob, start_index)
        # skipfirst = 0 if ((start_index[0] != end_index[0]) or (start_index[0] != sp and end_index[0] != sp)) else 1
        # print('21', start_index[0], sp)
        for i in range(beam):
            for j in range(beam):
                prob = start_prob[i] + end_prob[j]
                si, ei = min(start_index[i], end_index[j]), max(start_index[i], end_index[j])
        # Replace answer if calculated probability is larger than previous windows
                if prob > max_prob and (si == ei):
                    pass
                    # print('21', si, sp)
                if prob > max_prob and (fsp < si < lsp and fsp < ei < lsp) and ei - si <= 40:
                    max_prob = prob
                    # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                    answer = data['pid'][k], (data['offset'][k][si - fsp - 1][0]), data['offset'][k][ei - fsp - 1][-1]
                    # answer = tokenizer.decode(data['input_ids'][k][si : ei + 1])
    
    return answer

def post_proc(answer):
    answer = answer.replace(' ','')
    answer = answer.replace(',', '')
    tok = answer
    l, r, w, z, = 0, 0, 0, 0
    for e in tok:
        if e == '《':
            l += 1
        elif e == '》':
            r += 1
        elif e == '「':
            w += 1 
        elif e == '」':
            z += 1
    if len(tok) and tok[0] == '《' and tok[-1] != "》" and l > r:
        tok = tok + "》"
    if len(tok) and tok[-1] == '》' and tok[0] != "《" and l < r:
        tok = "《" + tok
    if len(tok) and tok[0] == '「' and tok[-1] != "」" and w > z:
        tok = tok + "」"
    if len(tok) and tok[-1] == '」' and tok[0] != "「" and w < z:
        tok = "「" + tok
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return tok
