# from rectpack import newPacker, PackingBin, GuillotineBssfSas, MaxRectsBl, \
#     MaxRectsBaf, MaxRectsBlsf, SkylineBl, SkylineBlWm, SkylineMwf, SkylineMwfl, \
#     SkylineMwfWm, SkylineMwflWm, GuillotineBssfLas, GuillotineBssfSlas
#
#
# def setup_cell_lengths(g1_list, g2_list):
#     rectangles = []
#     bins = []
#     for g2_len in g2_list:
#         bins.append((g2_len, 1))
#
#     for g1_len in g1_list:
#         rectangles.append((g1_len, 1))
#
#     packer = newPacker()
#
#     # Add the rectangles to packing queue
#     for r in rectangles:
#         packer.add_rect(*r)
#
#     # Add the bins where the rectangles will be placed
#     for b in bins:
#         packer.add_bin(*b)
#
#     # Start packing
#     packer.pack()
#
#     g2index2g1_lens_list = {}
#     g2_index = 0
#     g1_len_list = []
#     for abin in packer:
#         rect_width_list_by_bin = []
#         for rect in abin:
#             g1_len_list.append(rect.width)
#             rect_width_list_by_bin.append(rect.width)
#         g2index2g1_lens_list[g2_index] = rect_width_list_by_bin
#         g2_index += 1
#     #print(g2index2g1_lens_list)
#
#     g2index2g1_lens_int_index = {}
#     cursor = 0
#     for g2_id, g1_lens in g2index2g1_lens_list.items():
#         sum = 0
#         for l in g1_lens:
#             sum += 1
#         g2index2g1_lens_int_index[g2_id] = [cursor, cursor+sum-1]
#         cursor += sum
#
#     g2index2g1_lens_sum = {}
#     cursor = 0
#     for g2_id, g1_lens in g2index2g1_lens_list.items():
#         sum = 0
#         for l in g1_lens:
#             sum += l
#         g2index2g1_lens_sum[g2_id] = sum
#         cursor += sum
#
#
#     return g2index2g1_lens_int_index, g2index2g1_lens_sum, g2index2g1_lens_list, g1_len_list
#
# #print(setup_cell_lengths([2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], [21,21,22]))