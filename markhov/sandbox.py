import markhov
import numpy as np
from importlib import reload



bigrams = {'a':{']':0.25,'a':0.25,'b':0.5},
           'b':{']':0.25,'b':0.25,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }


# markhov chain of operations
# when you go to Merge, you also make a move in the bigram FSM
# when you go to copy, you make a copy of the buffer and append it to the string and buffer
# when you go to clear, you clear the buffer
ops = {'mg':{'mg':0.8,'copy':0.1,'clear':0.1},
         'copy':{'mg':0.3,'copy':0.2,'clear':0.5},
         'clear':{'mg':1.}
     }


# log em up
for a in bigrams:
    for b in bigrams[a]:
        bigrams[a][b]=np.log(bigrams[a][b])

for a in ops:
    for b in ops[a]:
        ops[a][b]=np.log(ops[a][b])



markhov.generate(bigrams,ops)


s = "[ a a b ]".split(" ")

s= "[ a a ]"

parses = [ [['['],[],['mg'],False] ]
for i in range(1,len(s)):
    print (i)
    print len(parses)
    if s[i] in bigrams[s[i-1]]:
        #print(parses)
        for parse in parses:
            print("parse to merge to: %s"%parse)
            parse[0].append(s[i]) #string
            parse[1].append(s[i]) #buffer
            parse[2].append('mg') #list of ops
            print ("merged parse: %s"%parse)
    for parse in parses:
        print("parse: %s"%parse)
        if len(parse[1])>0 and parse[0][-1] != ']': # if buffer not empty and sentence not done
            if not parse[3]:  #i if last special op not clear
                new_parse=[[],[],[],[]]
                new_parse[0]=parse[0][:] # copy the string
                new_parse[2]=parse[2][:] # copy the list of operations
                new_parse[1]=[] # clear the buffer
                new_parse[2].append('clear')
                new_parse[3]=True
                print ("new parse: %s"%new_parse)
                #print ("old parse: %s"%parse)
                parses.append(new_parse) # add this new parse

            print ("buffer ",parse[1])
            print ("copy? ", s[i+1 : i+1+len(parse[1])])
            if parse[1]==s[i+1 : i+1+len(parse[1])]:
                new_parse=[[],[],[],[]]
                new_parse[2]=parse[2][:]
                new_parse[0]=[s[:i+1+len(parse[1])]] # skip ahead to end of copy
                new_parse[1]=parse[1][:]+parse[1][:]
                new_parse[2].append('copy')
                new_parse[3]=False
                print ("new parse copy: %s"%new_parse)
                parses.append(new_parse)


parses = [ [['['],[],['mg'],False] ]
for parse in parses:
    print("parse: %s"%parse)
    i=len(parse[0])
    if i<len(s):
        if len(parse[1])>0: # if buffer not empty
            print ("\n Clear")
            if not parse[3]:  #i if last special op not clear
                new_parse=[[],[],[],[]]
                new_parse[0]=parse[0][:] # copy the string
                new_parse[2]=parse[2][:] # copy the list of operations
                new_parse[1]=[] # clear the buffer
                new_parse[2].append('clear')
                new_parse[3]=True
                print ("new parse: %s"%new_parse)
                #print ("old parse: %s"%parse)
                parses.append(new_parse) # add this new parse

            print ("\n Copy")
            print ("buffer ",parse[1])
            print ("copy? ", s[i+1 : i+1+len(parse[1])])
            if parse[1]==s[i+1 : i+1+len(parse[1])]:
                new_parse=[[],[],[],[]]
                new_parse[2]=parse[2][:]
                new_parse[0]=s[:i+1+len(parse[1])] # skip ahead to end of copy
                new_parse[1]=parse[1][:]+parse[1][:]
                new_parse[2].append('copy')
                new_parse[3]=False
                print ("new parse : %s"%new_parse)
                parses.append(new_parse)

        print ("\n Merge")
        if s[i] in bigrams[s[i-1]]:
            #print(parses)
            for parse in parses:
                print("parse to merge to: %s"%parse)
                parse[0].append(s[i]) #string
                parse[1].append(s[i]) #buffer
                parse[2].append('mg') #list of ops
                print ("merged parse: %s"%parse)




agenda = [ [['['],[],['mg'],False] ]
parses = []
while len(agenda)>0:
    for parse in agenda:
        print("\nparse: %s"%parse)
        i=len(parse[0])
        if i==len(s):
            parses.append(parse)
            agenda.remove(parse)
        else:
            if len(parse[1])>0: # if buffer not empty
                print ("\n Clear")
                if not parse[3]:  #i if last special op not clear
                    new_parse=[[],[],[],[]]
                    new_parse[0]=parse[0][:] # copy the string
                    new_parse[2]=parse[2][:] # copy the list of operations
                    new_parse[1]=[] # clear the buffer
                    new_parse[2].append('clear')
                    new_parse[3]=True
                    print ("new parse: %s"%new_parse)
                    #print ("old parse: %s"%parse)
                    agenda.append(new_parse) # add this new parse

                print ("\n Copy")
                print ("buffer ",parse[1])
                print ("copy? ", s[i : i+len(parse[1])])
                if parse[1]==s[i : i+len(parse[1])]:
                    new_parse=[[],[],[],[]]
                    new_parse[2]=parse[2][:]
                    new_parse[0]=s[:i+len(parse[1])] # skip ahead to end of copy
                    new_parse[1]=parse[1][:]+parse[1][:]
                    new_parse[2].append('copy')
                    new_parse[3]=False
                    print ("new parse : %s"%new_parse)
                    agenda.append(new_parse)

            print ("\n Merge")
            if s[i] in bigrams[s[i-1]]:
                parse[0].append(s[i]) #string
                parse[1].append(s[i]) #buffer
                parse[2].append('mg') #list of ops
                print ("merged parse: %s"%parse)



