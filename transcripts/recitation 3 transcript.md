
Guy Assa: So.
0:02
Guy Assa: who wants to remind us what we did in the previous lectures or stations, like, what topic we talked about. Decision 3. And before linear regression. Okay? And this is what you did last lesson right yesterday.
0:05
Guy Assa: Yeah, so logistic is part of part of the classifiers. So we will do it all all today.
0:22
Guy Assa: Okay, so today, we're going to talk about 3 algorithms. One is perception. The second one is a a list mean square that we will just present, and we won't dive into
0:31
Guy Assa: deep into it. But it's basically the same idea as the 1st one. And then we will talk about logistic regression in depth. We will understand it side to side, and I will. As you asked, or the group before you asked, I will present you an exam.
0:45
Guy Assa: And we'll do it like offline. I want to call it, because it's unofficial. So if you're here, you're lucky, I will show you one exam. We will go over it briefly. We're not going to stop in every
1:06
Guy Assa: a close and talk about what we're going to do there, because there's a lot of material that you haven't learned yet.
1:18
Guy Assa: but you will see it, and you understand and everything. So this would really be it.
1:25
Guy Assa: Okay? So just a little bit of a algebra reminder. So we have. In algebra we have hyper plan. What is a hyper plan is a subspace, as they mentioned is one less than of the ambient space. Right? So if we have, for example, one line, so the hyperplanes is a point. If we have
1:30
Guy Assa: 2D, as we have. So we have a linear line, and if it's it's 3D, so we have a plane, and so on and so forth, and we can continue so. This is a hyperplane. Does it mean that it's 1 less than Airbnb space like, why is it one less 2D. We have a line.
1:55
Guy Assa: It's because in 3D you have. It's the the shape is 3D, and here it's it's just like in 2D, here you have only X and y, you don't have another dimension to it.
2:13
Guy Assa: And
2:31
Guy Assa: yeah, it depends on the
2:36
Guy Assa: variables. Okay, so how we redefine it. So we said, it's N minus one from the space we are working working on. And we know the the equation for it. So we solve it. So for a linear line, we have only x to the one x, 1, which equals to some something
2:38
Guy Assa: and so on. So forth. So just to bear in mind that each one of these, it's like data point. Okay, this, this vector is a data point of the X and each X. It's like a feature. So we have height and weight and shoe size, and so on, and so on, and and I can plug it into the equation, and then it predicts something about this instance.
3:00
Guy Assa: And so, of course, we can do it in every space. The space depends on the features, and we can do it in every space we want.
3:21
Guy Assa: And so the X are the coordinates. And then the hyper plan. So what we get is a hyper plan. So let's say we have a line in 2D. So we have a line, a hyper plan that separates the space into 2 halves. Right? We have one half and the second half.
3:28
Guy Assa: And this is the task that we want to do now. Okay, we want to look at all the points that they are bigger than B, or if I'll remove if I move B here, so it would be 0 here. So all the points that are put on the equation, and they're bigger than 0 or B, and all the points that they're smaller. And if we have a similarity, so we can handle it in some way. But let's discuss here for bigger and smaller.
3:46
Guy Assa: Okay, so for example, let's talk about lines. So we have this line and this line, this line, all of them are going through the.
4:14
Guy Assa: and we can add some variable that we call it B, or data 0 in our case. And then we can. The line can can shift, and this line separates the space, this 2D space into 2 halves. And now I want to use the linearity to classify right? So I'll say I have linear line. This is the linear classifier. But now I don't want to predict the value of something I just want to say if it's
4:24
Guy Assa: 0, or one, if it's blue or red, if it's a black or white or something. So I want to separate the space into 2 parts, and I'll say everything that it's above. It's in one class, and everything that is below is the difference in the other class. Okay, we're talking here about
4:53
Guy Assa: 2 classes on. Okay. So this is the task that we want to do
5:09
Guy Assa: to do to to have some linear line or separator to separate the space into 2 2 sections, and then I can class
5:13
Guy Assa: die in the in this case, if I here here, for example, we have the one which is theta 0. Right?
5:25
Guy Assa: So I can say for this line, for example, what is above the what? The line is greater than one, and what is
5:33
Guy Assa: below the line is smaller than one in this equation. But in this equation I'm talking about in compared to 0. Because I move it's the same. Yeah, it's the same. It's the same, but it depends. If I'm saying it's bigger than one here, or bigger than 0 here. Yes, but in the other hand, one is like is down below. It's not like above. If there is a point there above the line, like in x 1.
5:42
Guy Assa: If I have, for example, the point
6:11
Guy Assa: 5 here and 10 here. Okay, so I can plug 5 and 10, and I'll get 4. So 4 is bigger than 0. So it means that 4 is somewhere here that 5 and 10 is somewhere here. So if it's above the line, I will classify it as one. For example.
6:14
Guy Assa: right? So this is my task. And this is what I want to do here. And of course, I have different lines.
6:34
Guy Assa: etc, etc. Okay, so we want to find the linear separator that if the result is greater than 0, as we saw now, I will classify it as plus one or minus one. It doesn't matter the class. I'm giving it because the class have no meaning. Because if I want to classify people that they have hard like.
6:38
Guy Assa: have some disease. Okay? So disease is plasma. One or or minus one doesn't matter right? So the class.
7:01
Guy Assa: it doesn't matter just for me to separate them. So if someone that have this disease I will give him plus one, and if someone else have this disease, I will give him plus one and the other one I will give minus one. If I want to switch them, I can switch them. But I just need to be consistent
7:09
Guy Assa: with with the labeling. Okay, so and again, if it's lower than 0. So it would be minus one or the opposite from the one I gave to the greater than 0.
7:26
Guy Assa: Okay, so what we need, basically, we need some hyper plan with weights. Right? We need some weights. Including the bias that will help me to to draw this hyper plan. And then I need to predict for each point for each data point. Rather, it's bigger. If I'm plug in
7:36
Guy Assa: it into the equation, if it's greater than 0, it will be one, and if it's smaller than the 0, it will be minus one or the opposite, it doesn't really matter. Just be consistent.
7:59
Guy Assa: So let's look at an example. So we have a Boolean function that coordinated. And right? So it's and it's in the it has to be positive positive in the 2
8:10
Guy Assa: entries. So this only this point will be labeled as positive, or one and the other one will be minus one. So what is the solution? Again, I want to have a line that will separate the space. So, for example, this line separate space and theta 0 is a 1 and a half right somewhere here, and then theta one is one, theta, 2 is one. I can draw this line.
8:22
Guy Assa: and this will separate perfectly. The space over the the samples in the training right into the training. I see it, and I and I can classify it perfectly.
8:50
Guy Assa: It confused me yesterday with the logistic, but I believe it will be the same here. How do I choose the direction in which
9:06
Guy Assa: what is one? What is minus one? It's something that I need to get like in the question from the questions. Let's say blue is positive, so blue is one. So I will go to the. It doesn't really matter. It's really matter on the
9:14
Guy Assa: when I'm trying to to evaluate, it will have an effect on the way that the line will converge because for example, if we start in line like this. So if you choose this to be positive, so it will move like this, and if we choose it to be negative. It will choose like it will move in the other direction. But it
9:33
Guy Assa: it doesn't really matter. It will be the same. But the way it will get there won't be the same, because the directions will be like opposite. I see. Okay, so it's just how many iterations that we need to do exactly if you will choose it in one way, you will have to do X and iterations, and the other way
9:55
Guy Assa: a half, plus x plus or half x or something, but you can't know it in advance, because you have no idea
10:17
Guy Assa: another function. It's the or function. So just one of the entries has to be one, and then it's labeled as plus one. And if you have both entries 0, so you'll get negative or minus one. So here, what is the solution. So I can take again theta one to be one, theta, 2 to be one.
10:24
Guy Assa: and theta is 0 to be 0 point 5, and I have this line, and then I separate it perfectly. We will talk about it later about
10:48
Guy Assa: how I can put here infinite lines right? I can put here many lines as I want. But this one is a good line for me, and I can stop here. And this will classify my problem perfectly.
10:56
Guy Assa: The vector that I want to do.
11:10
Guy Assa: It's not this one in particular to this slide is what
11:14
Guy Assa: if I try to look on the the website to the looks like
11:22
Guy Assa: again. I'm not sure if you want to to see if it's so, let's call X to y, because this is how we it's we learned in it. So I say, it's y equals to minus x 1 plus the unit. Vector
11:35
Guy Assa: again, you need the weight vector
11:53
Guy Assa: he's talking about the weights. Victor.
11:58
Guy Assa: the white selector. What about the white selector
12:01
Guy Assa: which has been the collapse of our line itself?
12:04
Guy Assa: This line is exactly this line. Do you agree? The it's we have the slope of mine of minus one, and it cuts the the Y or X 2 in. Let's say it's 0 point 5, or maybe I'll put it a little bit here, and then it would be 0 point 5. And then this is exactly this. It's the same. I just replaced X 2 with the
12:09
Guy Assa: another one multiplied by x 1 plus omega, 2 multiplied by x 2, then the weights vector
12:33
Guy Assa: this, if you want to measure the distance, what is plus and what is minus, and because it's the norm vector is perpendicular to the.
12:44
Guy Assa: to the, to the line. And then you know that the plus is in on the direction of the up. And yeah, but I don't have to. I can just plug in the numbers, because if I plug here 0 and 0. So it will be 0 0, and then it will be smaller than 0. So I will predict it to be negative one, because it's smaller than 0, it will be minus 0 point 5. The
13:00
Guy Assa: the you're talking about the distance that you want to calculate from the
13:27
Guy Assa: basically, you're correct. I can also
13:33
Guy Assa: like calculate the distance between of them. But
13:36
Guy Assa: here, in this method, I can also like predict, by just plugging the number there, and
13:40
Guy Assa: just take the the point, plug it in and see if it's great or smaller.
13:46
Guy Assa: So I don't know what you did in the in the lecture itself.
13:54
Guy Assa: Okay. So now we'll talk about the perceptron. So the perceptron is doing this exactly. So we'll understand how it
13:59
Guy Assa: how how it become the algorithm. So percept one is, of course, a binary classification algorithm, part of the supervised learning based on the future algorithm that we will learn. It's the base of a future algorithm that we learn, and a little bit of more motivation. So if some of you know, neural networks of deep learning so percept one is a single layer. So in in neural networks we have many layers.
14:07
Guy Assa: usually. So if you look at single layer, so single layer can be perceptron. The process that we're doing here can be sometimes perceptron. So perceptron is one of the most basic layers that we have. We have much more advanced layers. We can do, but one of them is perceptron, and the process that we're doing here is can be one of the layers in neural networks. So for single networks we have several
14:35
Guy Assa: process of perceptron. But if we look at one process, it's a singularity.
15:02
Guy Assa: So this is also in motivation. And it's good to know it for future material.
15:06
Guy Assa: Okay? So we know what we are looking for. We're looking for a linear separator, and we need to to find it so how we will find we'll start with random weights, and then we will improve the line according to some method, and as long as we have all, and we will talk about it in a minute. But what is the error? Is the output minus the target? The target is something I know, plus one or minus one. What is the output? The output is according to my line that I draw.
15:14
Guy Assa: Is it minus one or plus one? If it is, if it's above, it's something, and if it's below. It's something. So if they're both the same, both.
15:43
Guy Assa: I said that under the line they're both need to be minus one. So if they're both the same, so it will be minus one minus minus one, it will be 0. So you don't have an error, but if they're not the same, it can be, or that this is one, and this is minus one. So they're not the same. So the the result will be 2, or this is minus one, and this is one, and then it will be minus 2, right? So the error can be 2 or minus 2 in our case.
15:53
Guy Assa: So, for example, if we have this line, this is the line we started with, we just initiate some random data.
16:19
Guy Assa: and then we draw this line. So of course, we can see that we have error with this one. So we want to improve it somehow. So we want to shift the line a little bit little bit like this, and to have a perfect separation. So how we do it, what? What is the target that we want to? How we will improve the tote.
16:29
Guy Assa: Remember that all we have or we can play with. It's the theta right? Because the the values of the X are fixed, and then I have only the theta of the line that we can shift and modify it. So I want to modify it. How I will modify it!
16:48
Guy Assa: I will look, as we said about the target. The target here is red. Let's say red is minus one, and the output. The output is, I'm plugging in the point into the equation, and then I see if it's below the line, so I give it minus one. If it's above the line, give it plus one. So here I have, or 0, because they're both the same, or that I have 2 or minus 2, right? If they're not the same.
17:05
Guy Assa: I'll multiply it by the value of the X and multiply it by some learning rate, because I don't want to.
17:30
Guy Assa: It's similar to the learning rate in the gradient descent that we did, because I want. I don't want to improve the line too drastically
17:39
Guy Assa: or not in too little step, because I can improve it a little bit if it was here. So if I'll take the learning rate is too small, so the steps will be really, really small, and it will take me a lot of time to convert to here, and if the learning rate will be too large, it might jump all over, and I won't be able to capture this? Exactly. Okay. So in the process.
17:48
Guy Assa: well, here's the link, this one. Yeah. So I'll show you the, can you remind us the name of this? This letter.
18:10
Guy Assa: Yeah, I agree later. Later. Okay, thank you.
18:19
Guy Assa: Yes, I am
18:23
Guy Assa: so. So as we have here, we have the learning rate here that I multiply all the all this equation. I'm running all over all my data points. Okay, I'm summing all the data points and what I'm doing here. I'm calculating if it's correct, so it's 0 so and that doesn't care. And if it's not 0, if it's not the same, so it's 2 or minus 2, multiply by the value of X.
18:25
Guy Assa: And then I have here some sum of all over the old points, the old data points multiplied by the learning rate, and then I'll shift my theta according to this one. Okay, we'll see it in a minute. An example. It's very similar to linear regression.
18:52
Guy Assa: So yes, but we here we have. 1st of all, we don't have the process of
19:08
Guy Assa: take the derivative, and so on, and so forth. And you you have here labels like. So it's similar in somewhat. But the process it's it's different.
19:14
Guy Assa: But yes, we are improving the data as we're improving the
19:26
Guy Assa: when you say all of the data points, all of the feature of one sentence. All of the data points you're running all over the data points and calculating
19:30
Guy Assa: yesterday. Tell us that we fix after each.
19:41
Guy Assa: This is a stochastic, and we'll do it in a minute.
19:45
Guy Assa: But in the perceptron you're running on all over all points and calculating the theta according to all points.
19:48
Guy Assa: but we'll see in a minute an example of stochastic, and we'll do exactly what you said. 1 point.
19:55
Guy Assa: So it's similar to what we've learned in the linear regression. You're taking all the points. It takes much more time to do it, because you need to calculate each for every each point. If you have millions, so it will take you more time. But stochastic, it's faster, but is more noisy because you're changing the data according to 1 point, so might won't be exactly correct. Yes, here I have. X.
20:04
Guy Assa: The I is the under of X, and the D is the upper. Now d is the the sample. And I is the feature. Yes. So I'm changing data one.
20:30
Guy Assa: theta one corresponding to x 1. So here I'm putting the value of x 1.
20:45
Guy Assa: It's different. It's different annotation than yesterday. Yesterday. It was it was the d was I? And I was d, so it's exactly so. Here you're summing all over the data points. So this is a data point. And here the annotation of the data is I. So it we will see an example in a minute, and it will be much clearer.
20:51
Guy Assa: So I'll go over to you. It's not really important. It may be a little bit complicating things, but it's not that complicated. So what is the algorithm? So we are initiating some random. And then we repeat, until convergence, what is convergence? No, ao, okay, no, Ao, it's convergence. And what we're doing. So for each data point, we're calculating the sum, the sign.
21:12
Guy Assa: excuse me, plus one or minus one. And then for each data. We're updating it.
21:36
Guy Assa: We're taking the previous value, the previous value and adding the delta. What is the delta is exactly the equation that we saw right now. And this is all over all data points.
21:42
Guy Assa: Yes. So we will see it now on stochastic. Okay. Instead of doing it over all points, we'll do a stochastic example, and it will be much clearer. So here, instead of summing all points, we will do just a single point, and we will see how it looks like. So
21:56
Guy Assa: the only change from stochastic to the perceptron is here. Instead of summing all of them all the points, I will do it over 1 point and update it according to 1 point. So now we will see an example which is over 1 point. But it's the same idea.
22:13
Guy Assa: There is blue, red, and the and the initial guesses
22:30
Guy Assa: go over. Yeah, so it will be 0. The here it will be 0
22:35
Guy Assa: if you guessed perfectly so it's perfect.
22:44
Guy Assa: Prevent this case. But in this case we can
22:55
Guy Assa: get 0. If the initial guess is incorrect. Yeah, how you can, how you can know the guess
22:59
Guy Assa: if you have, if you can print it. Of course, if you have, if you have a 15 features.
23:06
Guy Assa: what's your guesses? You have 15 features and 1 million samples. What's your guess? What will you guess?
23:15
Guy Assa: 42, one. So in this, in this example, it's easy. Of course, I can draw the line here. Yeah, it's it's easy. I finish. But in this case that we get, we get 0 because we take every time the square.
23:23
Guy Assa: But in this case we take only the
23:42
Guy Assa: so- so here it's not an arrow
23:45
Guy Assa: in the sense of the distance you're talking about, because we calculated the distance in Msa.
23:51
Guy Assa: you want to prevent a situation that you're calculating this distance and will cancel each other. But here, I'm not asking.
23:58
Guy Assa: I'm not summing all the distances I'm just asking you're in your place or not, and then there won't cancel.
24:06
Guy Assa: Yes, and let's example. In left bottom, the opposite
24:22
Guy Assa: and the left bottom here. Yes, the opposite one, and the left top minus one.
24:31
Guy Assa: because everything is mistake here, and in the summary we will get 0.
24:39
Guy Assa: No. Here this this is not classified correctly. And I want to draw this line. So
24:45
Guy Assa: let's see this example, and and maybe
24:53
Guy Assa: you won't have. You won't have a result in this specific case that you said, because you want to do minus one upstairs and you don't. It's it's the sword. Let's see the example. And it will be much
24:55
Guy Assa: a guy very soon inside of the table that you showed previously some table plus one and minus one.
25:09
Guy Assa: The line 3. I didn't. I didn't got. Why, why it's become
25:16
Guy Assa: this one. Yes, why, it's here in in 3. Why, it's decreased.
25:22
Guy Assa: Because, the output. It's it's I. I will show I don't want to get too deep into it, because it might be confusing. But if this is a plus one, and this is minus one, so if I plug it in here. So it's plus one minus minus one. So it plus 2 right and the x is positive, right? And here I have minus. So all of this would be minus. So
25:27
Guy Assa: it will decrease the data in order to get closer to a but let's see the example. It will be much clearer. So I have these points. Okay, these 3 are classified as plus one. And this one is minus one. Again, it doesn't really matter
25:53
Guy Assa: if this is minus one. This is plus one or the opposite. It only will change the how the line will convert, but it will convert to the same point instead of like, maybe if all this plus one, so it will do like this. And if there were minus one, so it will do like this. Okay, so it doesn't really matter, but it will get to the same
26:07
Guy Assa: to the same position. One sample. Yeah.
26:26
Guy Assa: okay, so we have the weights because we guess them just randomly, theta 0 plus one, theta one minus 2.2, theta 2 0 point 1 5, and our leveling rate is 0 point 0 5. Okay, so this is the how we started. And of course we can see that it's not classified correctly. So we want to do some process to classify it correctly. So, of course, in the eye we can see that we need to do it differently. But we want to
26:31
Guy Assa: do some process that will do it by itself. So
27:00
Guy Assa: we will start stochastically. So we're taking just 1 point. Okay, I'm taking 1 point. This, this is the 1st point I'm taking.
27:04
Guy Assa: It's 1 minus one minus one. And the label is plus one. Okay. So now I want to know what's the sign. So I'm plug it in into the equation. Okay. So I can see that minus one minus one, I'm getting 0 point 1 5. It's above 0. It's above the line, and it's great. So I determined that above the line is plus one. So it's above the line great. So the output is plus one. The target is
27:11
Guy Assa: is one. And if I'm looking at the equation. So it's 0. So I don't need to. I don't need to update anything great. Now, I'm taking the second point. So I have minus one plus one. And again, when I'm plugging in, I'll get 0 point 4 5 greater than 0. So my output is plus one, and the target is plus one. Everyone is happy, everything is good now
27:41
Guy Assa: the second one. So I have the 3rd point. Excuse me, I have plus one minus one this point the target is plus plus one right? And if I'm plugging it into the equation, I will get minus
28:10
Guy Assa: point 25 right, which is smaller than 0. So my output will be one. So now it's not the same right, basically, because this is plus one. And my and I have minus one. So the line is not perfectly so. Now I need to change everything, change all the alphas. Okay. So there you have a reminder. So what I'm doing. I'm taking the previous data, which was
28:25
Guy Assa: 0 point 1. And now I'm updating it. How I'm taking the learning rate which we, it's a hyperparameter that we set it 0 point 0 5, multiply it by what? By the output minus the target right, which will be minus 2. Multiply by the x. So when we're talking about theta 0, we don't have X, so we're taking one because we don't have any coordinates for theta 0, because we
28:50
Guy Assa: it's our imaginary or bias feature. But if I want to update data one, so what will I do? I will take the last data data, one which was was minus 0 point 2,
29:18
Guy Assa: multiply it by the learning rate the output minus the targets, the output minus the target, and multiply it by the value of x 1, which is this, plus one, and I'll get
29:32
Guy Assa: minus 0 point 1. I'll do it the same for the less feature, for the less data. Excuse me, and here, I'll multiply it by X 2, the value of X 2, and I'll get this number. And now I can update the data right now, instead of this equation, I will have a little bit of a different equation.
29:45
Guy Assa: this equation, right? And this is over. This is stochastic. Why? Because I took only single point if I wanted to do it not stochastically. So I had to sum up all the points and then change the data and sum up all the points and change the data, then the change might be more mild. But
30:05
Guy Assa: you know advantage, disadvantage. We talk about it both stochastic and non stochastic. Okay, so now this is the line, and now it's correctly classified. By the way, if we took different learning rate, if it was much smaller, so the line might move from here to here, and it's not enough. So we need to maybe change the learning rate or do more iterations or something else. Yes.
30:27
Guy Assa: the line can both change location. The vector can both change location and change the angle as well. I think it's a simulation. No, the simulation. It moved the angle as well. I just didn't see it, because it's a small change
30:53
Guy Assa: plus one minus one. The point. The 3rd point that we checked.
31:20
Guy Assa: Now I've been up to the top. The pressure is active.
31:27
Guy Assa: where is that on the graph? Which one are we talking about? Plus one minus one?
31:31
Guy Assa: Yeah. I colored it to to be in compliance to this. But maybe it's a little bit. So now we're taking another point plus one plus one.
31:39
Guy Assa: Do we normalize the volume early stuff, these values
31:51
Guy Assa: which value yes, they will. They will move because you're multiplying this one. This is a disadvantage of this method. If you have outliers, it will throw you away so you can't normalize it, because this is
31:59
Guy Assa: there is.
32:21
Guy Assa: So each case will jump.
32:22
Guy Assa: Yeah. But remember that if you normalize it, you're you're not changing the the variance. It will stay instead of being like
32:28
Guy Assa: this point and one here, it will be now
32:34
Guy Assa: like this, but it will be in the same ratio. So it's not really
32:38
Guy Assa: important to normalize it. But this is a problem with this. We don't fix.
32:43
Guy Assa: It can't really help with outliers like this stochastic and then non stochastic. Yes, it's a problem in this case. So if we're taking this case. So we see that this point is minus one. We can obviously see that it's not classified correctly. We plug it in, and we see
32:48
Guy Assa: for fact that they don't have the same class. So we need to change them, how we change them. We're updating it the same manner. Okay. So we doing the same equations we've seen before. And now we have new datas. And now we have new lines.
33:05
Guy Assa: And now this points classified, incorrect. So we need to continue and iterate. And if we're getting to this point, so we'll classify it as well. And we're finished with this line. Okay, after I have no arrow I can finish. So maybe it's not the best line, because maybe I have a red point here.
33:20
Guy Assa: and then this line should shift a little bit here. But for this data I finished because I get the perfect notification. And this is also a problem with prescription that it's sometimes not giving me the best result, because
33:39
Guy Assa: like, you might say, maybe we want to to be. The distance from this and from this would be the minimum. But we will talk about later in Svm. Which, doing exactly the same. But here, as I found a line that classified perfectly, I'm stopping and said, Okay, it's perfectly classed.
33:52
Guy Assa: And now, if I want to predict a new point. So just give me the x 1 and x 2. I'll plug it in. If it's greater than 0, it will be red. If it's smaller it will be blue. That's it. And the label just important. It's red and blue. It's not minus one and plus one.
34:09
Guy Assa: Okay, so this is important to bear in mind.
34:25
Guy Assa: The label is red and and blue. Yes. Can you go back just a little bit? I just want to make sure what's the second line there the Sgm
34:27
Guy Assa: gross
34:36
Guy Assa: the sign. This is the sign of the yeah. The result is minus. So the my sign would be minus. Okay, okay, one more question we previously spoke. But is there are there any cases where you can't get
34:38
Guy Assa: a line that gives you a error of 0 with this? Red, let's say in the bottom left corner. Exactly. So, this is the next slide. So the problem with this
34:51
Guy Assa: the problem with this algorithm exactly is that if we have data that is not linear classified, that I can't classify it linearly. For example, red
35:00
Guy Assa: dot here, so it will not convert. It will always try to to find the line. But but I don't have. So this algorithm is not good for
35:11
Guy Assa: not linearly separable. We will talk about methods that we can make data linear separable in the future. But for now, if we have this problem with red Dot here, it's impossible. I'm giving up, and I'm not. I won't use perceptron to do it, because it's impossible to classify just different than linear regression, because they were looking for the minimal error. Yeah. And you can stop wherever you want, but here we will stop it.
35:23
Guy Assa: Perfect, perfect. 0. Yes, exactly so. What we can do we can use Lms exactly for it, and we won't dive into an example. But just understand the the idea. So instead of giving a sign to my output, I'm giving the value.
35:52
Guy Assa: Okay. And now, we were taking a different approach, little bit of a different approach. So we want to. Instead of classified correctly, we want to minimize minimize the all. Okay? So how we would do it.
36:09
Guy Assa: So our goal here now is not to find a a separable hyperplane with no arrow but to minimize the distance. Okay? And we'll see illustration in a minute. So we want to. This is our task. So if we're looking at is so we have.
36:26
Guy Assa: This is the we're looking at distances. So now, I want to minimize the distance. Okay, so this is will be my distance formula. So how it looks like in the perfect scenario. Okay, all. Now, I'm separating the data to the positive points which I'll call them, plus one. And the negative points which I'll call them minus one. Okay? So in perfect scenario.
36:45
Guy Assa: remember that the O is the line right in the oh, I have data 0 plus data one x 1, and it will give me the output. So in the perfect scenario I will get, I will get all
37:11
Guy Assa: all my output here will be one. And then this would be 0, all of them and all of the what should be minus we get minus in my algorithm. And this would be 0, then I have 0 plus 0. So the the entire thing will be 0. Right? So this is the best case scenario
37:24
Guy Assa: when everything is classified correct. But if it's not classified correctly so here I'll have something different than one. And here I'll have something different than minus one. And then I can calculate the distance for this equation.
37:43
Guy Assa: Okay, so basically, what I want to do here is so what
37:58
Guy Assa: the illustration of what we're doing here? It's like drawing 2 lines. So this is my equation.
38:03
Guy Assa: And now I'm drawing 2 lines. And now I'm asking how far these points from this line and how far the red points from this line. This is exactly what we're doing here. We're separating into like 2 sub problems. And we want to find the line that minimize the
38:09
Guy Assa: the distance of this line, this one and this points from this line. Okay, this is the task that we want to do. This is what's happening behind the scene. Okay, we're doing the same process as before. But this, what's happening behind the scenes and we won't go. We won't show an example. But this is what capturing in Lms, okay, so now we can get, for example, this line.
38:27
Guy Assa: okay, or different line the best line will be. We will see it in a minute.
38:51
Guy Assa: But we will get a different line and that we can have some arrows. Okay, we can have maybe the best. Maybe this is the best line, so we can have arrows, because this will be the line and this will misclassify, and this will be misclassified. But maybe this is the best line that we can get.
38:55
Guy Assa: I don't know forever. Okay, so we're doing basically the same as we did before, but because we're not not looking at a sign. And we we're handling with distance. Now we can shift the problem a little bit and have some arrows and look for the minimum distance. Okay? So there is no any. There's no example or something. But this is the notion that we can get from the Lms.
39:12
Guy Assa: Okay.
39:39
Guy Assa: And here, can you repeat that this is the best thing?
39:40
Guy Assa: You calculate the error as in the equation, that we've seen that in this equation.
39:52
Guy Assa: that the output will be a number, and then you plug it here, and then you update the data according to the number and behind the scene. What happened is exactly this. You're asking how far the blue dots
39:59
Guy Assa: from this line, and how far they read from this one?
40:12
Guy Assa: Why, why, we there is no divided by the amount of sentence
40:16
Guy Assa: it's a you can divide, but it's it's not really important, because it's a constant. So you can.
40:22
Guy Assa: But you can divide by the by, the by, the number of instances.
40:28
Guy Assa: And okay.
40:33
Guy Assa: so the differences between person and Lms, so they're both handling with the target value 0 1 or plus one, the output calculation. So here we're looking about the sign. And here we're looking about the real value of the function itself. Optimization. So here we're looking for 0 classification a 1 here we're looking for minimum distance.
40:36
Guy Assa: and if it converge so, the Lms will always converge because it can control it by the number of iterations, so we'll tell him, do 10,000 iterations and stop, and it might have errors, but it will converge, and here, in the supply it will converge only if it's a linear sufferable.
40:58
Guy Assa: The separation within the perception optimum, if it's separable and the Lms not have to be, say and optimum. And the perception is not sensitive to outside, because it always separated perfectly, and the Lms, of course, will have errors. It depends on how we
41:15
Guy Assa: telling it to be okay. So the limitations we've seen it. We talk about it. And the Lms is not optimized as well. So what we we can do if we have other methods. Yes, we have logistic regression. Okay? So now we will move to logistic regression. So
41:34
Guy Assa: we learn about the perception we understand. We learn about the Lms. We understand the idea of of what change we have and what we benefits we get.
41:53
Guy Assa: And now we will get to logistic regression from linear regression. So I will explain you why linear regression is not the best method
42:02
Guy Assa: to use, and we will shift from linear regression to logistic regression. Okay. So don't try to understand why we're losing why we're using a linear regression to this problem. Just let's see the flow and understand why we step by step how we're getting there. So
42:12
Guy Assa: let's remember the linear regression problem. So we want to predict value. And we have continuous values of Y, and we have a set of features. And then we said, Let's do a linear function. This is our hypothesis function. It's some linear function with theta and the vector, weeks. And we have the cost function that we're saying, we're looking about the Mse mean square.
42:28
Guy Assa: We're just plugging in the X point in the equation minus the target. And then we're getting the error. And we want to minimize it to be as small as possible. And this is the the process that we're doing
42:54
Guy Assa: so now we want to do it for classification, right? We don't want to get the value. We want to get one and 0. So why, it's not good approach to use this method, because if we're doing it, for example, for this data. So here I did a linear regression of this data. So what will be the problem? So this is the line. I fit it to the data. So this is the best line. And now can I predict 0? And one. So
43:07
Guy Assa: the points they don't have to be all 0 and all one, just an illustration like the problem. It's easy to to see.
43:35
Guy Assa: So what's the problem here? How will I classify this point? What will it be?
43:43
Guy Assa: 0 1? So I guess it will be one right, because it's close to close to the, to the red ones. But what will be this point will be
43:50
Guy Assa: blue or or red.
43:58
Guy Assa: No idea. So what would can help me to decide.
44:01
Guy Assa: What should I do? What should I add? And maybe I'll finish.
44:05
Guy Assa: maybe, but I can maybe just add a threshold right? I can say above
44:10
Guy Assa: 0 point 5, it will be red, and below it will be blue. Right? So maybe maybe it's enough. Right? I can say, okay, so let's do for classification. Let's take the linear line, and I'll say that everything that is below 0 point 5. It would be one, and everything is below it would be 0. And then I'm finished. Maybe I don't need logistic regression. I can just predict
44:15
Guy Assa: a linear thing with this setup. So this is enough. It's not enough.
44:35
Guy Assa: Let's look at this example. For example, in this world, if I'm taking the linear line and putting a threshold of 0 point 5. So this point will classify as red because it's above the line, so it would be red. But if I have this example, for example, now, the line would be
44:41
Guy Assa: tilt over a little bit like this, because I have some point that dragging it to be more in a, the slope will be lower. So now this point will classify if I need to. To project it over the line now it's
44:57
Guy Assa: it will be under 0 point 5. And now it's blue. So this prop this method of linear regression with threshold. It's problematic because it's sensitive to outliers. So I want other method that won't be sensitive to outliers and will help me to classify everything correctly.
45:13
Guy Assa: Okay. And this is another example of outliers. So I have here
45:32
Guy Assa: the the data is pretty much classified
45:37
Guy Assa: nicely. So I have the same accuracy of linear regression and logistic regression. But once I have an outlier like here, so the linear regression accuracy, it's also somewhat good. 0 point 9 1. But the logistic regression is much better
45:39
Guy Assa: 0 point 0 2
45:56
Guy Assa: points above. So I want a better method that will handle the outliers better. So we're moving from linear to logistic to get to be better on the outliers and basically better than most of the things.
45:59
Guy Assa: So how we will do it. So we said that linear line is very sensitive to outlier. Can I go back one second through a slide?
46:17
Guy Assa: It's something that I'm not so sure about, how does the logistic regression can be
46:24
Guy Assa: drawn as a linear linear line? I'm just trying to think because the logistic regression is, it's a separator, but it's not a separator that it's linear, right? So we will see in a minute how you get to the logistic regression from the linear regression, so you can draw the line of the the logistic regression is a linear. Yeah, okay? So I will wait. So maybe visually, it won't be good because you're saying, this
46:31
Guy Assa: doesn't look really good. It's just a visualization. And it's not a
47:01
Guy Assa: okay. That's what. So we want to transfer the data. So basically, we want to the logistic regression. You will understand in a minute why we want to do it. But we want to get, instead of infinite space between minus infinity and infinity that the linear regression can get. We want to predict values between 0 and one.
47:16
Guy Assa: Okay, because we want it to be probabilities. Okay, we want to change it to probabilities, and we'll understand in a minute why. So now we want to shift it.
47:39
Guy Assa: and to some probabilities. So I can say that I will represent
47:47
Guy Assa: the function as a probability. Okay, this is what I want to represent, that my hypothesis function will be a probability function. Okay, this is what I want to to resolve to plug in the X and get some probability. Okay? Why, it's not enough.
47:52
Guy Assa: This can scale, as we said, the the values of X can scale between minus infinity to infinity. And we want the the probability to be bound between 0 and one. So we need to transform P somehow somewhat to make it as a probability. So
48:09
Guy Assa: it's a little bit of a maybe complicated notion. But let's
48:27
Guy Assa: try to understand what we're doing here. So we are starting with this prediction. And as we said, we want to shift the P to be probability. So we'll take the odds. Okay, what is the odds where we will divide the P in one minus P. Now, we're making it as sorts. Okay? So I just want to
48:34
Guy Assa: somewhat plug in X's and get probability. This is what I want to to get and see in a minute, and how we drive from this one to the probability. So this is the steps
48:55
Guy Assa: and how they find it. So now, if we're doing this, you decided that beta 0 and plus beta one x equal to P.
49:05
Guy Assa: Why is why is this decision? This is the equation, and I want it to be represented as probability
49:16
Guy Assa: at the end. Okay, I want to represent it somewhat as a probability like it's you will see in a minute the end result, and you will understand it. And this is the way we we got to this point. So I just want to show you how we got there. So if we're dividing by one minus P, so the output can be between 0 and infinity. But as we said, we want between 0 and one
49:23
Guy Assa: so we'll take the log of the odds. Okay, so if we're taking log of this.
49:48
Guy Assa: And now we will take P out because we want to isolate. P. So we'll get the sigmoid function. Okay, so here we get the sigmoid function which it's between 0 and one. We'll see it in a minute. Why? And basically the linear function that we started here is here. Okay? So basically, I can take
49:53
Guy Assa: the linear, my linear word and plug it in here, and then it will shift to probability. Word, okay, to do probabilistic word
50:13
Guy Assa: which lies between 0 and one, if it's possible to repeat it, because something the algebra doesn't make sense for me. How do I move from P. To p, 1 minus p, and that to log P without changing the results. The beta 0 and beta one
50:25
Guy Assa: without changing the yeah, because we're changing only one side of the the function. It's just interpretation. It's not. You're not changing here, here, not changing anything. You want to
50:45
Guy Assa: do some mathematical process on this side. If I do do it, I don't need to do it both sides. That's the
50:56
Guy Assa: because the representation of this one will be equal to this one. So you just want to know this function.
51:04
Guy Assa: You can say, like, I want to put it square. And it's like a squaring the yeah. But then I need to square both sides. I can't square your equation. Is this one you don't. This is a representation of the data. It's not. You're not changing something here inside, you're not changing the X or the, but they're just a representation of it. So if you have, for example, a line, so you can say I want to represent it as as like this. So just put everything in
51:11
Guy Assa: in square. So it's not changing anything here. If you just represent it in a different way. You you shifted the the function. But you haven't changed the points itself just in a different space. It's like you're taking it to another space.
51:38
Guy Assa: So it's like, okay, I have more example if you want. But okay, so we just.
52:00
Guy Assa: And it doesn't make it can take long it will be there. So we're just taking the function and do some things on the function can isolate our variable. And then we can get the sigmoid function, what's the meaning of it? So if we look at this one? So, okay, so this is the function. This is how it looks. Now, it's not a linear line. Now, now it's a squiggled line called sigmoid and sigmoid. It's 1 of the most important
52:08
Guy Assa: function that you use a lot of in deep learning and everything. So it's it's good to to know it. This equation are the same.
52:36
Guy Assa: And what we basically did, we took infinite space. Okay, which is this equation. And we did something to the equation, some process that we've seen. And we turn it to this equation to to this one. Okay, so we just
52:45
Guy Assa: divided by something, put a log and isolate the P and everything. And we got this one. Okay. So we didn't change the point itself. They still have values here in this one, and we can still represent this this line in the linear space if you want to shift it. But we will do other calculation here. So we in somewhat in some sense, we moved from linear space to sigmoid space. Okay, so this is what we did so we can now forget
53:01
Guy Assa: for for a minute this one. Now, we handling
53:30
Guy Assa: this problem. Okay, so we have here our X and the Beta or the theta in our example. And and we can learn things and do things over this space. Okay, so this is the I just want to show you how we moved from
53:32
Guy Assa: the linear space to the sigmoid space. Okay, which still have still remember, derived from the same place? Right? So this is the process we did. And just to convince you that this is really lies between 0 and one and its probabilities, and we'll see in a minute why we wanted probabilities. So the X value are from the real world or from the world. We start in the beginning, and here it will be the sigmoid score, so it will take 0.
53:48
Guy Assa: So if you put here 0, it would be 1 1 divided by 2. It's half.
54:16
Guy Assa: and it's good, because it's here right half of the plane is a
54:21
Guy Assa: here itself. So it's it's a good point to be in the middle if we take minus infinity. So here, minus infinity would be plus infinity, and then one plus infinity. It's infinity. So one divided by infinity would be 0, and plus infinity would be all the way around will be one and in between, we have values in between 0 and one, and so on and so forth. So we can
54:26
Guy Assa: convince ourselves this lies between 0 and one. And now we will see why we wanted to do it.
54:50
Guy Assa: Okay, so now, we would say that, okay, we have this, of course, and now I can
54:57
Guy Assa: project the the points on this graph, but I still don't know what will be the value
55:06
Guy Assa: if you remove the red line. So how I will determine a point that it's here. So I need a line to to decide if it's above or below, and if it's a red or blue. So we still, we need still some pressure to help us determine. But now we will see in a minute. This process is much more accurate. Okay, so now, the logistic regression classification will be that my hypothesis function
55:13
Guy Assa: is that I'm using the sigmoid instead of the linear function. Okay, now I'm using a sigmoid, and I have a threshold as a decision. If it's above threshold, which here it's 0 point 5, it would be one, and if it's below, it would be 0. Okay. So now, for example, I want to predict a new point.
55:38
Guy Assa: So I have this data set. And now a new point. Enter now, I project it on the sigmoid, and it's below. It's above 0 point 5. So I'll predict it as a red. Okay, so this is the red at all. Because, yeah, you project the the point on the sigmoid, and it will be here, and it's above the 0 point 5, and then.
55:57
Guy Assa: and you will project it because it set a threshold. Okay, so but before we had a linear line that the linear Heim was really influenced by values that they were here. But here, because of the sigmoid it will like
56:23
Guy Assa: If Laura, it will will help us with the really outliers and extreme values say, the difference.
56:41
Guy Assa: like the error, is above 0 point 5.
56:51
Guy Assa: The error is about 0 point 5. So the question we're asking here is, what is the probability of you being one? So here it will be 0 point 7. So it has 0 point 7 probability to be one. So if it's above 0 point 5. I will say, Okay, you are one. If it was 0 point 2. So I said, No, your probability to be one is 0 point 2. So you're probably
56:54
Guy Assa: 0. Okay, so this is the question I'm asking to determine which one.
57:16
Guy Assa: So okay, I understand now, we have a different function. It's not linear. So if you didn't understand the process, we did. So just take it as it is, this is the function we're looking at function that they are in the shape of sigmoid. And now we want to understand which.
57:21
Guy Assa: Okay, so this is one function. This is one shape. But we can have many shapes. Right? I can have this line looks like this, or like this. So which one is better. So we need a process to understand
57:38
Guy Assa: some cost function that will tell me how good is this line and some method to improve from line one line to another. Okay, so
57:50
Guy Assa: even if you understand how we got to this line or not, we will take this line, this function as as a
57:59
Guy Assa: is true. And now we will do the same as we we did before. We will want to learn
58:06
Guy Assa: this line and to have a cost function. So if we have a set of points. So here it's not organized in some way. It's just set of points that they have. Let's say, heights for each one, and it's labeled like, maybe boys and girls or something. So if we would project it on some
58:12
Guy Assa: linear line. This is this is the 1st line that we guess right? We guess Theta and we have this line, and then we can project it or transform it to the sigmoid. Right? So I transform it to the sigmoid, because in the sigmoid function. I have this linear line, right? Because I have the
58:31
Guy Assa: in the power of the EI have this sign. So in this case, they're both classifying. Classify the data correctly. So everything is good. But it's not always the case.
58:50
Guy Assa: For example, we can have a sigmoid that on my 1st guess, my 1st guess here on the data that I have will draw me this sigmoid, and I want to somewhat improve, maybe perfectly classified. This one, or or maybe I can have a better solution. So I want somehow to move from this one to a better one, from this one to this one or so. I want I want to have also cost function and also a method to minimize the cost function.
59:03
Guy Assa: So the cost function will be maximum likelihood. So the cost function that we haven't yet learned what maximum likelihood is. But let's take it as a given now, and we'll understand it later, and the method to improve it will be guardian descent, as we did in the face recitation. Okay, so we'll see it in a minute, and when minimizing the cost, the effect will be to move
59:32
Guy Assa: the sigmoid. Yeah, it will. It will, or change the number. The F. That we took the threshold. What would it change the the threshold? It's something you set before as a hyperpower metric. You said the threshold is 0 point 5, or if you want to accept more false, positive, or something, so we can. Secondly, with the threshold, it will change the shape of the sigmoid, and then you will. We will see it in a minute.
59:53
Guy Assa: Okay, so what is the likelihood? The likelihood is simply taking each point and
1:00:21
Guy Assa: just put it in the equation of the sigmoid and get the number. So we said that the number are relying between 0 and one, and I'm looking at each point as independent. This is the reason I'm multiplying, and I want to maximize the likelihood
1:00:28
Guy Assa: a maximum likelihood estimation. This is the name of this method, so I will take all the blue points which I know they're blue, because this is the training. I will take them as they are, because I want them to be one. Right? So I want to maximize the likelihood, the maximum number that the point can get is one the maximum probability. So I'll take the blue one, the blue dots
1:00:43
Guy Assa: as as they are, because I want them to be one and the red dots. I want them to be 0. So I take them my one minus the result because the optimum
1:01:05
Guy Assa: the optimum line will separate them in the optimum place. All the blue dots will be on the one, and all the red dots will be on the 0 right, the optimum line. And then I'll get one multiple, one multiple, one multiple one and
1:01:16
Guy Assa: one minus 0 multiple one minus 0 and so on and so forth. And then I will get maximum likelihood. Okay, this is the reason I'm doing the one minus. Okay, to get the maximum likelihood. So for example, this point is 0 point 9. So this will lower my likelihood because I'm taking a 1 minus this point. Okay, so this is.
1:01:30
Guy Assa: that was clear why I'm doing the minus one.
1:01:54
Guy Assa: and as in many, for for the the calculation to be more easy, we're taking the log. So we're simply doing the log as here. So as before, we're just taking the log. And then this is the log likelihood. This will be our cost function metric.
1:01:57
Guy Assa: Okay? And just to understand why we can do it. So if we took if we look about on this graph. So if I'm taking the function of minus log X,
1:02:16
Guy Assa: so I'm taking the minus log X for the labels that are labeled one, and the probability that I will give them will be as close to one, so the cost will be as close to 0. And the reason I'm taking one minus for the 0 is because the function is look like this, and if I will predict them a probability that it's close to 0.
1:02:25
Guy Assa: The the cost function will be close to 0, and if I'm summing them together so the cost will be as many of them as possible. So it's exactly like we've seen here. But this is the logical mathematical logic behind. Why, we can do
1:02:47
Guy Assa: the log, and why, we're taking one miles. Okay, so this is what we want to do. And this is our cost function. Simply you. When you have a line.
1:03:02
Guy Assa: you have this squiggles line, you're taking it. You're projecting the points on it, and if it's blue, if it's positive, take it as it is, and if it's negative
1:03:12
Guy Assa: one minus, okay? And then you have cost function. You can simply add it together. And you have a number. And you want to minimize it. Okay, so now we will see
1:03:21
Guy Assa: the mathematics steps. But it's really straightforward. So we'll go over it. So if we're looking at a single point.
1:03:32
Guy Assa: Okay, so if this is the Y is one. So this will be one. This will be 0. And then we're looking at the function itself. And if it would be 0. So this would be one. And we're looking about one minus the points, exactly what we've seen now. And this is exactly what we just said. Right, if it's 0. So we're looking about this function. And if it's 1, we're looking at the function itself.
1:03:39
Guy Assa: it's all the same. What we said is in different notations. And now, if I want to take all the points together, so I need to multiply all the points together, to do it for each point, to get the sum of the entire data right to get the the cost of the entire data and not the same single point. So now we have, the maximum likelihood, the log maximum likelihood of the entire data.
1:04:04
Guy Assa: And then I can separate it to these 2 components because we said it, all the positive points will look like this, and all the negative points are close to 0 when we're looking at this shape. And now it's again just Al- algebra. So don't be afraid. So we're taking the land.
1:04:29
Guy Assa: As we said, the log, the log, the maximum log. So we separate the multiplication to a summation of of the land. And then we take the power outside this step. And and always in machine learning, we want to look at minimization and not maximization. So we're putting minus. And then it's minimum. Okay, we take this problem, put minus and then look in it as a minimum.
1:04:46
Guy Assa: Okay from this equation that it's called binary cross entropy. Not that important for you
1:05:14
Guy Assa: we plug in because this was the hypothesis function right? Some hypothesis function. But now we're plugging in the sigmoid, because this is the problem that we're talking about. So we're plugging in our sigmoid, which looks like this. And then, if I'm taking the land with the land rules. So I'm getting this equation. And here I'm doing some algebra and getting this equation.
1:05:21
Guy Assa: you can go over it. It's not something complicated. And finally, when I open the brackets and summing everything, we will get this line, which this is the cost. Okay, so we look at it here. This is the cost. And we know everything right? Because y, it's 0 or one, it's the label. This is the line, the linear line, right data and the X
1:05:45
Guy Assa: and minus one of one plus e to the power of the line, so I can plug each point and get the cost
1:06:09
Guy Assa: when I sum them all together, and this will be my cost. Now, if I want to improve myself, or to see where should I improve? I need to do gradient descent, as we did
1:06:16
Guy Assa: always right.
1:06:28
Guy Assa: Take the derivative according to data one, for example, calculated the derivative according to theta, 2, so on and so forth, and change the data according.
1:06:30
Guy Assa: So the process of value and descent would be the same. But instead of Mse, here we had Mse. Before. Now we have the binary. So this is this is the difference. Yeah, just one question about the data. When we use the data on the above function, it's a matrix of all the
1:06:41
Guy Assa: of other pro-probabilities.
1:07:02
Guy Assa: What is the metrics, the the values of the data transform
1:07:04
Guy Assa: the T is transformed. Right? For the matrix. Yeah, it's just annotation. It's not. It's just that you will. If you have a vector, of X that you can multiply it by a vector, of
1:07:09
Guy Assa: of data. Oh, okay, so vector, of X, yeah, because here, we're talking about the entire data. Yeah, that's why I understood, I wanted to make sure that the T is for the transformed matrix of data.
1:07:20
Guy Assa: Taking one data. And then you have single data here.
1:07:38
Guy Assa: It's similar to linear regression in some sense. But instead of linear function, we have sigmoid function, and instead of Mse cost function, we have binary cost entropies. But the entire process is the same. So even if you didn't understand nothing from what I said. If you understand linear regression, this is similar to linear regression. We have sigmoid function instead of linear function. We have different cost function, but the process is all the same.
1:07:58
Guy Assa: Okay, so you can just plug in the things here, and everything is right. And if you ask, what's the how? You can interpret visually the line of logistic regression. So you just take the data from here and you can interpret it so it won't be
1:08:27
Guy Assa: exactly the classification, but you can somehow.
1:08:42
Guy Assa: as you like to interpret it as a linear line.
1:08:46
Guy Assa: And this is will be the. This is, we will end the process. So, as we said, it's exactly the same like linear regression. We have the sigmoid function. We have different cost function than before, a little bit complicated, but not too much. And the goal is to minimize this one. Okay? And then if we finally
1:08:49
Guy Assa: if we finally find our
1:09:09
Guy Assa: our sigmoid, that is the best right in linear regression where we find the best line. So here we find the best sigmoid.
1:09:13
Guy Assa: And now we need to decide the threshold right with hyperparameter. So let's say 0 point 5. And now, if we have a new point, and this is the best signaling for our data, because it results the smallest
1:09:22
Guy Assa: cost function. So now, if I have a new point, I just see where it relies on. The X relies here. Great.
1:09:36
Guy Assa: here it's above 0 point 5. So the probability of this point to be equal to one is 0 point 9, then I will classify it as one. Okay, so this is how I classify new point.
1:09:43
Guy Assa: So basically, we could just eliminate all what we learned. And just look at this. And if you understand this, you understand logistic regression. This is what you need to know. But we I tried like to explain you how you get from this one to this one, and I hope some of this
1:09:57
Guy Assa: makes sense, in some in some sense
1:10:14
Guy Assa: and final thing I want to say about it before we will move on is that we can classify.
1:10:17
Guy Assa: We learn how to how to classify 2 classes. But if I have, for example, 3 classes.
1:10:24
Guy Assa: so I can shift it to a binary problem. So I'll take each class and say, Okay, one predictor. Each one of them will be different. Sigmoid. Okay, one will be, or it can be other things. But let's say we're talking about sigmoid. So
1:10:29
Guy Assa: this I will do the process when I'll ask. I'll give one to if you're A and else 0 if you're B or C, then I'll do a new process, which I'll call it Predictor B, which will get one if it's B and 0 if it's else.
1:10:44
Guy Assa: And the same process for the predictor of C, and then I can now have. Now I have 3 process, different process, and they'll have different
1:11:00
Guy Assa: shape of sigmoid, for example. And now I can take the point, the new point, which is on the same, the same XX 5, and see where it falls in each one of them, and then I'll take the greatest, for here the great. The greatest result is 0 point 7, so I'll take I'll take it as a V, so I can shift the 2 class problem into multi-class problem. Okay, this is how I can shift.
1:11:09
Guy Assa: This is a what. So no, this is not a soft phone. This is a prediction of
1:11:36
Guy Assa: Multi class in a binary way. Let's say so. Here, I'm just
1:11:42
Guy Assa: taking the problem and and do it 3 times each time for a different class. And then I'm taking in this example the the one that is most probable under this exception. Okay. So this is, this is just
1:11:50
Guy Assa: and
1:12:04
Guy Assa: like a bonus thing x 5 is some new point that you're seeing. And you want to predict it. And you just plug it into the sigmoid and see where it gets the most probable
1:12:06
Guy Assa: great questions inside of the graph of the Sin Point going for the loss function.
1:12:23
Guy Assa: If it is on.
1:12:33
Guy Assa: So it means if I predict one, and the true result is one.
1:12:36
Guy Assa: So the the loss function is 0.
1:12:42
Guy Assa: So here. We are not saying what you are predicting here. We're using probabilities. So you will plug in
1:12:45
Guy Assa: here. You will plug in
1:12:53
Guy Assa: H. 0, which H. 0 is the sigmoid and the sigmoid. The result of it is number between 0 and one, so the loss function will be the summation of numbers between 0 and one.
1:12:56
Guy Assa: and it will be also between 0 and one.
1:13:12
Guy Assa: Yes, but it's a number between 0 and one, and it doesn't have to be one or 0, it could be any number in between.
1:13:17
Guy Assa: Okay, so for example, here, if I'm 0,
1:13:24
Guy Assa: yeah, the graph here. Yeah, it can go action.
1:13:30
Guy Assa: So here you have this line, which is the sigmoid. And now you're taking each point, and you put it inside the equation of the sigmoid. So this point will equal 0 point 2 9, and this point will equal to 1 9 1, and so on and so forth. You just put logo on each point, and you have a number. But I'm talking about the loss function. This is the loss function. This is exactly the loss function.
1:13:35
Guy Assa: Here. You get the number. And this is the the open us.
1:14:03
Guy Assa: And then, if you want to improve it, you're changing a little bit. The you're doing gradient descent. You change it a little bit if you improved. Great, if not.
1:14:08
Guy Assa: yeah.
1:14:18
Guy Assa: In this case, the multi class. Yes, this is a
1:14:19
Guy Assa: if it's 1. So it's a if it's 0, it's else.
1:14:31
Guy Assa: If it's a 1, it's B, and if it's 0, it's else. And if it's a 1, it's C, and if 0 is else. So
1:14:36
Guy Assa: yeah, you're there is no connection between. There is connection between all of them. Because here, if you if you're not A, you're B or C, and if you're not B, you're A or C, but yeah, but the process you're doing it independent.
1:14:46
Guy Assa: One room A and one room B, so
1:15:04
Guy Assa: say again, there is option. If it ran them separately. It can get results. For example, at 1 point
1:15:08
Guy Assa: got 100% for a and also 100% for B,
1:15:16
Guy Assa: maybe yeah, maybe 100%. It's less.
1:15:22
Guy Assa: And the prediction it can be. Yes, it can be both 0 point 9. Yeah. And then you need to decide. Maybe flip a coin or something.
1:15:26
Guy Assa: but always the the end result. Flip a coin. Yeah. But but
1:15:34
Guy Assa: basically you can, we just talk about a hypothetical way to shift the 3 class problem into 2 class problem. Because we take the, we have 3 class, and we're just saying this one is, or you one audio, A, or you not. This one is, or you, B, or you not, and or you see, or you're not, and then you're taking the point itself and see what is the projection of it on the sigmoid, and then you decide by the maximum. If you have 2 values the same, so
1:15:40
Guy Assa: you need to decide someone else, some other process.
1:16:04
Guy Assa: Great. So if you don't have any more questions, let's move to the
1:16:10
Guy Assa: that I want to show you.
1:16:15
Guy Assa: Just stop recording.
1:16:18
Guy Assa: Can I ask question the basic implementation of analysts, for example, they use this method, the amnist? Yes.
1:16:21
Guy Assa: each with a number.
1:16:33