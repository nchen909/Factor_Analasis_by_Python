n=int(input())
aaa=[]
wordtext=input()
for word in wordtext.split():
    a=int(word)
    aaa.append(a)
aaa=aaa.sorted(reverse=True)
print(aaa)