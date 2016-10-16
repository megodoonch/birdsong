


def copy_and_apply_transition(state,t):
    """ 
    Applies transition t in the given state,
    if this is actually possible given the string
    and buffer and bigram chain; if so, returns a copy of the state
    with the transition applied and the pointer 
    in the string advanced to the new location.
    If this transition is not possible, return None.
    """
    # TODO
    return state[:]
    


agenda = ["S"]

while len(agenda):
    task = agenda[0]
    (bis,buf,(qs,ops),k)=task # extract the current task

    # Now let's do the task...
    for t in all_possible_transitions(qs[-1]):
        newtask = copy_and_apply_transition( (bis,buf,(qs,ops),k),t)
        if newtask!=None:
            agenda.append(newtask)

    del agenda[0] # remove the task we just completed



agenda = thing::things

thing

contrinruebw9tihj bthingdss]
