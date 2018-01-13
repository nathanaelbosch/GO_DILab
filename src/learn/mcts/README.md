# Monte-Carlo Tree Search

The work so far consists of followint [this interesting tutorial from Jeff Bradberry](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/), and adapting it to GO. [Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) also has some explanations.

I made adjustments to the code from the tutorial, sampling not uniformly but according to some *Policy Network*, and by playing only to some max depth and then, if the game did not finish, evaluating the result using some *Value Network*.