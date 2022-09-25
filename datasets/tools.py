from random import randint, sample
import sys

import config


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).

    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    # Returns
        A list of words (or tokens).
    """
    if sys.version_info < (3,):
        maketrans = string.maketrans
    else:
        maketrans = str.maketrans
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):  # noqa: F821
            translate_map = {
                ord(c): unicode(split) for c in filters  # noqa: F821
            }
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = {c: split for c in filters}
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def get_intent(username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    intent = 'define intent ' + username + 'Intent:'
    if origin:
        intent = intent + ' from endpoint("' + origin + '")'
    if destination:
        intent = intent + ' to endpoint("' + destination + '")'

    for index, target in enumerate(targets):
        if target:
            if 'for' not in intent:
                intent = intent + ' for '
            intent = intent + 'client("' + target + '")'

            if index != (len(targets) - 1):
                intent = intent + ', '

    for index, mb in enumerate(middleboxes):
        if mb:
            if 'add' not in intent:
                intent = intent + ' add '
            intent = intent + 'middlebox("' + mb + '")'

            if index != (len(middleboxes) - 1):
                intent = intent + ', '

    for index, metric in enumerate(qos):
        if metric and metric['name'] not in intent:
            if 'with' not in intent:
                intent = intent + ' with '

            intent = intent + metric['name'] + '("' + metric['constraint']
            intent = intent + '","' + metric['value'] + '")' if metric['constraint'] is not 'none' else intent + '")'

            if index != (len(qos) - 1):
                intent = intent + ', '

    if start:
        intent = intent + ' start hour("' + start + '")'
    if end:
        intent = intent + ' end hour("' + start + '")'

    if allow:
        if allow not in intent:
            intent = intent + ' allow trafic("' + allow + '")'
    if block:
        if block not in intent:
            intent = intent + ' block trafic("' + block + '")'

    return intent


def get_entities(username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    entities = username
    if origin:
        entities = entities + ' ' + origin
    if destination:
        entities = entities + ' ' + destination

    for target in targets:
        if target:
            entities = entities + ' ' + target

    for mb in middleboxes:
        if mb:
            entities = entities + ' ' + mb

    for metric in qos:
        if metric:
            if metric['name'] not in entities:
                entities = entities + ' ' + metric['name'] + ' ' + metric['constraint']
                if metric['constraint'] is not 'none':
                    entities = entities + ' ' + metric['value']

    if start:
        entities = entities + ' ' + start
    if end:
        entities = entities + ' ' + end

    if allow:
        if allow not in entities:
            entities = entities + ' allow ' + allow
    if block:
        if block not in entities:
            entities = entities + ' block ' + block

    return entities


def write():
    with open(config.DATASET_PATH, 'wb') as file:
        for i in range(config.DATASET_SIZE):
            qos = []
            sampled_metrics = sample(config.DATASET_QOS_METRICS, randint(0, 4))
            for metric in sampled_metrics:
                sampled_constraint = sample(config.DATASET_QOS_CONSTRAINTS, 1)[0]
                while metric[0] is 'bandwidth' and sampled_constraint is 'none':
                    sampled_constraint = sample(config.DATASET_QOS_CONSTRAINTS, 1)[0]
                qos.append({'name': metric[0], 'constraint': sampled_constraint, 'value': str(randint(0, 100)) + metric[1]})

            username = sample(config.DATASET_USERNAMES, 1)[0]
            origin = sample(config.DATASET_LOCATIONS, 1)[0]
            destination = sample(config.DATASET_LOCATIONS, 1)[0]
            while destination == origin:
                destination = sample(config.DATASET_LOCATIONS, 1)[0]
            target = sample(config.DATASET_TARGETS, 1)[0]
            mbs = [mb for mb in sample(config.DATASET_MIDDLEBOXES, randint(0, len(config.DATASET_MIDDLEBOXES)))]
            start = sample(config.DATASET_HOURS, 1)[0]
            end = sample(config.DATASET_HOURS, 1)[0]
            allow = sample(config.DATASET_TRAFFIC, 1)[0]
            block = sample(config.DATASET_TRAFFIC, 1)[0]
            entities = get_entities(username, origin, destination, target, mbs, qos, start, end, allow, block)
            intent = get_intent(username, origin, destination, target, mbs, qos, start, end, allow, block)
            file.write(entities + ' > ' + intent + '\n')


def write_alt():
    with open(config.DATASET_PATH, 'wb') as file:
        for i in range(config.DATASET_SIZE):
            qos = []
            for metric in range(randint(0, 2)):
                qos.append({'name': '@qos_metric', 'constraint': '@qos_constraint', 'value': '@qos_value'})

            username = '@username'
            origin = '@location' if randint(0, 10) % 2 == 0 else ''
            destination = '@location' if randint(0, 10) % 2 == 0 else ''
            target = ['@target' for i in range(randint(0, 2))]
            mbs = ['@middlebox' for i in range(randint(0, 2))]
            start = '@hour' if randint(0, 10) % 2 == 0 else ''
            end = '@hour' if randint(0, 10) % 2 == 0 else ''
            allow = '@traffic' if randint(0, 10) % 2 == 0 else ''
            block = '@traffic' if randint(0, 10) % 2 == 0 else ''
            entities = get_entities(username, origin, destination, target, mbs, qos, start, end, allow, block)
            intent = get_intent(username, origin, destination, target, mbs, qos, start, end, allow, block)
            file.write(entities + ' > ' + intent + '\n')


def read():
    lines = []

    input_words = []
    output_words = []

    with open(config.DATASET_PATH, 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        if line and not line.startswith('#'):
            input_text, output_text = line.split('>')
            input_words.append(text_to_word_sequence(input_text, filters=config.DATASET_FILTERS))
            output_words.append(text_to_word_sequence(output_text, filters=config.DATASET_FILTERS))

    return input_words, output_words


def read_split():
    lines = []

    fit_input_words = []
    fit_output_words = []

    test_input_words = []
    test_output_words = []

    with open(config.DATASET_PATH, 'r') as f:
        lines = f.read().split('\n')

    fit_lines = sample(lines, int(len(lines) * 0.7))
    for line in fit_lines:
        if line and not line.startswith('#'):
            input_text, output_text = line.split('>')
            fit_input_words.append(text_to_word_sequence(input_text, filters=config.DATASET_FILTERS))
            fit_output_words.append(text_to_word_sequence(output_text, filters=config.DATASET_FILTERS))

    test_lines = list(set(lines) - set(fit_lines))
    for line in test_lines:
        if line and not line.startswith('#'):
            input_text, output_text = line.split('>')
            fit_input_words.append(text_to_word_sequence(input_text, filters=config.DATASET_FILTERS))
            fit_output_words.append(text_to_word_sequence(output_text, filters=config.DATASET_FILTERS))

    return fit_input_words, fit_output_words, test_input_words, test_output_words


def deanonymize(intent, username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    intent = intent.replace('@username', username)
    intent = intent.replace('@location', origin, 1) if origin is not None else intent
    intent = intent.replace('@location', destination, 1) if destination is not None else intent

    for target in targets:
        intent = intent.replace('@target', target, 1)

    for mb in middleboxes:
        intent += intent.replace('@middlebox', mb, 1)

    for metric in qos:
        intent = intent.replace('@qos_metric', metric['name'], 1)
        intent = intent.replace('@qos_constraint', metric['constraint'], 1)
        if metric['constraint'] is not 'none':
            intent = intent.replace('@qos_value', metric['value'], 1)

    intent = intent.replace('@hour', start) if start is not None else intent
    intent = intent.replace('@hour', end) if end is not None else intent

    intent = intent.replace('@traffic', end) if allow is not None else intent
    intent = intent.replace('@traffic', end) if block is not None else intent

    return intent


def anonymize(username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    entities = '@username '
    entities += '@location ' if origin is not None else ''
    entities += '@location ' if destination is not None else ''

    for target in targets:
        entities += '@target '

    for mb in middleboxes:
        entities += '@middlebox '

    for metric in qos:
        entities += '@qos_metric ' + '@qos_constraint '
        if metric['constraint'] is not 'none':
            entities += '@qos_value'

    entities += '@hour ' if start is not None else ''
    entities += '@hour ' if end is not None else ''

    entities += 'allow @traffic ' if allow is not None else ''
    entities += 'block @traffic ' if block is not None else ''

    return entities.strip()
