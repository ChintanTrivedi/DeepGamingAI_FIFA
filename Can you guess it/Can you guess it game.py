-Can you guess it?-
import random
even=input('please enter your birth year :')
number=int(input())
if number<2004:
    print('sorry you cant play this game,better try next time!!! ')
else:
    print('cogratulations you can play this game') 
    winning_number=random.randint(1,10)
    guess=1
    guess_1=int(input('guess a number betweent 1 and 10 :'))
    game_over=False
    while not game_over:
        if number ==winning_number:
            print(f"you win,and you guesed this number in {guess} times ")
            game_over=True 
        else:
            if number<winning_number:
                print('too low :')
                guess += 1
                number=int(input('guess again :'))
            else :
                print('too high :') 
                guess +=1
                number=int(input('guess again :'))


    
        

