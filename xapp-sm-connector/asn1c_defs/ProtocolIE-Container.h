/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "E2AP-Containers"
 * 	found in "/home/sjana/ASN-Defns/e2ap-oran-wg3-v01.00.asn"
 * 	`asn1c -fno-include-deps -fcompound-names -findirect-choice -gen-PER -no-gen-OER`
 */

#ifndef	_ProtocolIE_Container_H_
#define	_ProtocolIE_Container_H_


#include <asn_application.h>

/* Including external dependencies */
#include <asn_SEQUENCE_OF.h>
#include <constr_SEQUENCE_OF.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct RICsubscriptionRequest_IEs;
struct RICsubscriptionResponse_IEs;
struct RICsubscriptionFailure_IEs;
struct RICsubscriptionDeleteRequest_IEs;
struct RICsubscriptionDeleteResponse_IEs;
struct RICsubscriptionDeleteFailure_IEs;
struct RICindication_IEs;
struct RICcontrolRequest_IEs;
struct RICcontrolAcknowledge_IEs;
struct RICcontrolFailure_IEs;
struct ErrorIndication_IEs;
struct E2setupRequestIEs;
struct E2setupResponseIEs;
struct E2setupFailureIEs;
struct ResetRequestIEs;
struct ResetResponseIEs;
struct RICserviceUpdate_IEs;
struct RICserviceUpdateAcknowledge_IEs;
struct RICserviceUpdateFailure_IEs;
struct RICserviceQuery_IEs;

/* ProtocolIE-Container */
typedef struct ProtocolIE_Container_1412P0 {
	A_SEQUENCE_OF(struct RICsubscriptionRequest_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P0_t;
typedef struct ProtocolIE_Container_1412P1 {
	A_SEQUENCE_OF(struct RICsubscriptionResponse_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P1_t;
typedef struct ProtocolIE_Container_1412P2 {
	A_SEQUENCE_OF(struct RICsubscriptionFailure_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P2_t;
typedef struct ProtocolIE_Container_1412P3 {
	A_SEQUENCE_OF(struct RICsubscriptionDeleteRequest_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P3_t;
typedef struct ProtocolIE_Container_1412P4 {
	A_SEQUENCE_OF(struct RICsubscriptionDeleteResponse_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P4_t;
typedef struct ProtocolIE_Container_1412P5 {
	A_SEQUENCE_OF(struct RICsubscriptionDeleteFailure_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P5_t;
typedef struct ProtocolIE_Container_1412P6 {
	A_SEQUENCE_OF(struct RICindication_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P6_t;
typedef struct ProtocolIE_Container_1412P7 {
	A_SEQUENCE_OF(struct RICcontrolRequest_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P7_t;
typedef struct ProtocolIE_Container_1412P8 {
	A_SEQUENCE_OF(struct RICcontrolAcknowledge_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P8_t;
typedef struct ProtocolIE_Container_1412P9 {
	A_SEQUENCE_OF(struct RICcontrolFailure_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P9_t;
typedef struct ProtocolIE_Container_1412P10 {
	A_SEQUENCE_OF(struct ErrorIndication_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P10_t;
typedef struct ProtocolIE_Container_1412P11 {
	A_SEQUENCE_OF(struct E2setupRequestIEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P11_t;
typedef struct ProtocolIE_Container_1412P12 {
	A_SEQUENCE_OF(struct E2setupResponseIEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P12_t;
typedef struct ProtocolIE_Container_1412P13 {
	A_SEQUENCE_OF(struct E2setupFailureIEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P13_t;
typedef struct ProtocolIE_Container_1412P14 {
	A_SEQUENCE_OF(struct ResetRequestIEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P14_t;
typedef struct ProtocolIE_Container_1412P15 {
	A_SEQUENCE_OF(struct ResetResponseIEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P15_t;
typedef struct ProtocolIE_Container_1412P16 {
	A_SEQUENCE_OF(struct RICserviceUpdate_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P16_t;
typedef struct ProtocolIE_Container_1412P17 {
	A_SEQUENCE_OF(struct RICserviceUpdateAcknowledge_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P17_t;
typedef struct ProtocolIE_Container_1412P18 {
	A_SEQUENCE_OF(struct RICserviceUpdateFailure_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P18_t;
typedef struct ProtocolIE_Container_1412P19 {
	A_SEQUENCE_OF(struct RICserviceQuery_IEs) list;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} ProtocolIE_Container_1412P19_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P0;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P0_specs_1;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P0_1[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P0_constr_1;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P1;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P1_specs_3;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P1_3[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P1_constr_3;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P2;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P2_specs_5;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P2_5[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P2_constr_5;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P3;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P3_specs_7;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P3_7[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P3_constr_7;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P4;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P4_specs_9;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P4_9[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P4_constr_9;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P5;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P5_specs_11;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P5_11[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P5_constr_11;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P6;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P6_specs_13;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P6_13[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P6_constr_13;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P7;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P7_specs_15;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P7_15[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P7_constr_15;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P8;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P8_specs_17;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P8_17[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P8_constr_17;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P9;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P9_specs_19;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P9_19[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P9_constr_19;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P10;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P10_specs_21;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P10_21[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P10_constr_21;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P11;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P11_specs_23;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P11_23[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P11_constr_23;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P12;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P12_specs_25;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P12_25[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P12_constr_25;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P13;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P13_specs_27;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P13_27[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P13_constr_27;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P14;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P14_specs_29;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P14_29[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P14_constr_29;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P15;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P15_specs_31;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P15_31[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P15_constr_31;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P16;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P16_specs_33;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P16_33[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P16_constr_33;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P17;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P17_specs_35;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P17_35[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P17_constr_35;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P18;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P18_specs_37;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P18_37[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P18_constr_37;
extern asn_TYPE_descriptor_t asn_DEF_ProtocolIE_Container_1412P19;
extern asn_SET_OF_specifics_t asn_SPC_ProtocolIE_Container_1412P19_specs_39;
extern asn_TYPE_member_t asn_MBR_ProtocolIE_Container_1412P19_39[1];
extern asn_per_constraints_t asn_PER_type_ProtocolIE_Container_1412P19_constr_39;

#ifdef __cplusplus
}
#endif

#endif	/* _ProtocolIE_Container_H_ */
#include <asn_internal.h>
