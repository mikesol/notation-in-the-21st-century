# Machine learning for the layout process

---

# What is machine learning

- Randomly generate differentiable equations with tens of thousands of variables.
- Find a local minimum of the equations based on fully-known input data.
- Use the local minimum to make predictions about partially-known data.

---

# Example - XOR (concept)

- XOR is a binary boolean gate that yields `true` when _either_ of the inputs are true and _false_ otherwise.
    - T T -> F
    - T F -> T
    - F T -> T
    - F F -> F

- Let's watch a machine learn XOR to get a sense of what machine learning is.

---

# Example - XOR (ML)

    !python
    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense

    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    target_data = np.array([[0],[1],[1],[0]], "float32")

    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['binary_accuracy'])

    model.fit(training_data, target_data, nb_epoch=500, verbose=2)
    print model.predict(training_data).round()


---

# Why is this called _learning_?

- This is kind of how we learn.
- Some networks of equations are inspired by neural networks in the brain.
- It sounds impressive.

---

# When does machine learning make sense?

- When there is no easily-grokable model underlying a process.
- When the process yields sensible results with few outliers.
- When there is a ton of data.

# Does machine learning make sense in musical layout?

- _Yes_.
- There is no single, unified model underlying musical layout.
- Classical engraving is hard to get right, but it is rarely shocking or unpredictable.
- There is lots of data - we know the context of elements in traditional engraving _and_ where they are placed.

# Case study - slurs

- Slurs are notoriously difficult.
- LilyPond has over 20 exposed and 100 internal variables it uses for slur layout, many of which are interdependent in ways we don't understand.
- While LilyPond usually produces a sensible outcome, no one really gets why or how it does this. In other words, its functioning is basically as opaque as machine learning.
- Slurs, in classical engraving, provide us tons of data. Using OKR, we can amass a corpus of millions of slurs' placement (_target_) and the objects influencing their placement (_input data_).

---

# Slurs in LilyPond

---

# How to make a slur dataset

---

# Possible algorithms

---

# Gotchyas

- Circular dependencies

---

# Further research

- I unfortunately don't have enough bandwidth in the near future to work on this.
- It seems that Dorico is thinking along these lines, although I haven't seen their code.
- Would be nice to see more research in this area, happy to advise a GSoC project if anyone is interested.