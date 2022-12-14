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
