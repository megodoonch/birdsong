bis = {'[':{}}
for s in corpus:
    s=['[']+s.split(' ')
    for i in range(len(s[1:])):
        bis[s[i-1]]=bis[s[i-1]].get(s[i],{})
        bis[s[i-1]][s[i]]=bis[s[i-1]].get(s[i],0)+1
