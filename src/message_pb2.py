# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='message.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\rmessage.proto\"\'\n\tPredicate\x12\x0b\n\x03str\x18\x01 \x01(\t\x12\r\n\x05nargs\x18\x02 \x01(\x05\"]\n\tCandidate\x12\x19\n\x05pred1\x18\x01 \x01(\x0b\x32\n.Predicate\x12\x19\n\x05pred2\x18\x02 \x01(\x0b\x32\n.Predicate\x12\x0b\n\x03rel\x18\x03 \x01(\t\x12\r\n\x05score\x18\x04 \x01(\x01\" \n\x04Rank\x12\x18\n\x04list\x18\x01 \x03(\x0b\x32\n.Candidate\"(\n\x04\x45\x63ho\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x13\n\x04rank\x18\x02 \x01(\x0b\x32\x05.Rankb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_PREDICATE = _descriptor.Descriptor(
  name='Predicate',
  full_name='Predicate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='str', full_name='Predicate.str', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nargs', full_name='Predicate.nargs', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=56,
)


_CANDIDATE = _descriptor.Descriptor(
  name='Candidate',
  full_name='Candidate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pred1', full_name='Candidate.pred1', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pred2', full_name='Candidate.pred2', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rel', full_name='Candidate.rel', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='score', full_name='Candidate.score', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=151,
)


_RANK = _descriptor.Descriptor(
  name='Rank',
  full_name='Rank',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='Rank.list', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=185,
)


_ECHO = _descriptor.Descriptor(
  name='Echo',
  full_name='Echo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='Echo.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rank', full_name='Echo.rank', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=187,
  serialized_end=227,
)

_CANDIDATE.fields_by_name['pred1'].message_type = _PREDICATE
_CANDIDATE.fields_by_name['pred2'].message_type = _PREDICATE
_RANK.fields_by_name['list'].message_type = _CANDIDATE
_ECHO.fields_by_name['rank'].message_type = _RANK
DESCRIPTOR.message_types_by_name['Predicate'] = _PREDICATE
DESCRIPTOR.message_types_by_name['Candidate'] = _CANDIDATE
DESCRIPTOR.message_types_by_name['Rank'] = _RANK
DESCRIPTOR.message_types_by_name['Echo'] = _ECHO

Predicate = _reflection.GeneratedProtocolMessageType('Predicate', (_message.Message,), dict(
  DESCRIPTOR = _PREDICATE,
  __module__ = 'message_pb2'
  # @@protoc_insertion_point(class_scope:Predicate)
  ))
_sym_db.RegisterMessage(Predicate)

Candidate = _reflection.GeneratedProtocolMessageType('Candidate', (_message.Message,), dict(
  DESCRIPTOR = _CANDIDATE,
  __module__ = 'message_pb2'
  # @@protoc_insertion_point(class_scope:Candidate)
  ))
_sym_db.RegisterMessage(Candidate)

Rank = _reflection.GeneratedProtocolMessageType('Rank', (_message.Message,), dict(
  DESCRIPTOR = _RANK,
  __module__ = 'message_pb2'
  # @@protoc_insertion_point(class_scope:Rank)
  ))
_sym_db.RegisterMessage(Rank)

Echo = _reflection.GeneratedProtocolMessageType('Echo', (_message.Message,), dict(
  DESCRIPTOR = _ECHO,
  __module__ = 'message_pb2'
  # @@protoc_insertion_point(class_scope:Echo)
  ))
_sym_db.RegisterMessage(Echo)


# @@protoc_insertion_point(module_scope)
